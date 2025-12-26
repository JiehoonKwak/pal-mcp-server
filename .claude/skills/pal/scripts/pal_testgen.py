#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=1.0.0",
#     "openai>=1.0.0",
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL TestGen - AI-powered test generation with comprehensive coverage.

Run with uv (recommended):
    uv run scripts/pal_testgen.py --files FILE [FILE ...] [options]

Usage:
    python pal_testgen.py --files FILE [FILE ...] [options]

Options:
    --files FILE [FILE ...]    Files to generate tests for (required)
    --prompt PROMPT            Additional instructions (e.g., "focus on edge cases")
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --thinking-mode MODE       Thinking mode: minimal/low/medium/high/max (default: high)
    --json                     Output as JSON

Examples:
    # Generate tests for a file
    python pal_testgen.py --files src/auth.py

    # Focus on edge cases
    python pal_testgen.py --files src/utils.py --prompt "focus on edge cases"

    # Generate tests for multiple files
    python pal_testgen.py --files src/api/*.py --thinking-mode max
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response, infer_confidence  # noqa: E402
from conversation import ConversationMemory  # noqa: E402
from file_utils import read_files  # noqa: E402

from config import load_config  # noqa: E402
from providers import execute_request, get_provider  # noqa: E402


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser(description="PAL TestGen - AI-powered test generation with comprehensive coverage")
    parser.add_argument("--files", nargs="+", required=True, help="Files to generate tests for")
    parser.add_argument("--prompt", help="Additional instructions (e.g., 'focus on edge cases')")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking mode (default: high)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()

        # Initialize conversation memory
        memory = ConversationMemory(config)

        # Get or create conversation thread
        conversation_history = ""
        if args.continuation_id:
            thread = memory.get_thread(args.continuation_id)
            if thread:
                conversation_history = memory.build_history(thread)
            else:
                thread = memory.create_thread("testgen", {"files": args.files})
        else:
            thread = memory.create_thread("testgen", {"files": args.files})

        # Load system prompt
        system_prompt = load_prompt("testgen")

        # Read files
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to generate tests for")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== CODE TO GENERATE TESTS FOR ===")
        user_prompt_parts.append(file_content)

        # Add custom prompt
        if args.prompt:
            user_prompt_parts.append("\n=== ADDITIONAL INSTRUCTIONS ===")
            user_prompt_parts.append(args.prompt)

        user_prompt_parts.append("\n=== TEST GENERATION REQUEST ===")
        user_prompt_parts.append(
            "Please generate comprehensive tests for the code above following the guidelines "
            "in your system prompt. Cover happy paths, edge cases, error scenarios, and "
            "integration points. Use appropriate test framework conventions."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Execute request with high thinking mode by default for thorough analysis
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=0.7,
            thinking_mode=args.thinking_mode,
            config=config,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Generate tests for: {', '.join(args.files)}" + (f"\n{args.prompt}" if args.prompt else ""),
            files=args.files,
            tool_name="testgen",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="testgen",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Infer confidence from response (high default for test generation)
        confidence = infer_confidence(
            response["content"],
            has_errors=False,
            has_warnings=False,
            has_actionable_items=True,
        )

        # Build agentic result with rich metadata
        result = build_agentic_response(
            tool_name="testgen",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence=confidence,
            files_examined=args.files,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output with agentic metadata
            print(f"\n{'=' * 60}")
            print("Test Generation")
            print(f"Files: {', '.join(args.files)}")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Thinking Mode: {args.thinking_mode}")
            print(f"Continuation ID: {thread['thread_id']}")
            print(f"Confidence: {confidence}")
            print(f"{'=' * 60}\n")
            print(response["content"])
            print(f"\n{'=' * 60}")
            if response.get("usage"):
                usage = response["usage"]
                print(f"Tokens: {usage.get('input_tokens', 'N/A')} in / {usage.get('output_tokens', 'N/A')} out")
            print(f"\nNext actions: {', '.join(result['agentic']['next_actions'][:2])}")
            print(f"Related tools: {', '.join(result['agentic']['related_tools'])}")
            print(f"{'=' * 60}\n")

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if args.json:
            print(json.dumps(error_result, indent=2))
            sys.exit(1)
        else:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
