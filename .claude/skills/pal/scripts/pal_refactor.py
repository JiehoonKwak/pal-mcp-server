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
PAL Refactor - Refactoring analysis with external AI models.

Run with uv (recommended):
    uv run scripts/pal_refactor.py --files FILE [FILE ...] [options]

Usage:
    python pal_refactor.py --files FILE [FILE ...] [options]

Options:
    --files FILE [FILE ...]    Files to analyze (required)
    --focus AREA               Focus area (codesmells, decompose, modernize, organization)
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --thinking-mode MODE       Thinking mode (default: high)
    --json                     Output as JSON

Examples:
    # Basic refactoring analysis
    python pal_refactor.py --files src/main.py

    # Focus on code smells
    python pal_refactor.py --files src/service.py --focus codesmells

    # Focus on decomposition opportunities
    python pal_refactor.py --files src/monolith.py --focus decompose
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402
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
    parser = argparse.ArgumentParser(description="PAL Refactor - Refactoring analysis with external AI models")
    parser.add_argument("--files", nargs="+", required=True, help="Files to analyze")
    parser.add_argument(
        "--focus",
        choices=["codesmells", "decompose", "modernize", "organization"],
        help="Focus area for analysis",
    )
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
                thread = memory.create_thread("refactor", {"files": args.files})
        else:
            thread = memory.create_thread("refactor", {"files": args.files})

        # Load system prompt
        system_prompt = load_prompt("refactor")

        # Read files
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to analyze")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== CODE TO ANALYZE ===")
        user_prompt_parts.append(file_content)

        # Add focus area
        if args.focus:
            user_prompt_parts.append("\n=== FOCUS AREA ===")
            focus_descriptions = {
                "codesmells": "Focus on code smells: long methods, deep nesting, duplicate code, complex conditionals, naming issues.",
                "decompose": "Focus on decomposition: breaking up large components, single responsibility violations, extracting modules.",
                "modernize": "Focus on modernization: deprecated patterns, language updates, library upgrades, removing legacy code.",
                "organization": "Focus on organization: file structure, naming conventions, module boundaries, import patterns.",
            }
            user_prompt_parts.append(focus_descriptions[args.focus])

        user_prompt_parts.append("\n=== ANALYSIS REQUEST ===")
        user_prompt_parts.append(
            "Please analyze the code above for refactoring opportunities. "
            "Identify issues by severity (CRITICAL > HIGH > MEDIUM > LOW) and provide "
            "specific suggestions with effort/benefit analysis."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Execute request with high thinking mode for deep analysis
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
            content=f"Analyze for refactoring: {', '.join(args.files)}"
            + (f" (focus: {args.focus})" if args.focus else ""),
            files=args.files,
            tool_name="refactor",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="refactor",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build agentic result with medium confidence (needs validation)
        result = build_agentic_response(
            tool_name="refactor",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence="medium",
            files_examined=args.files,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output with agentic metadata
            print(f"\n{'=' * 60}")
            print("Refactoring Analysis")
            print(f"Files: {', '.join(args.files)}")
            if args.focus:
                print(f"Focus: {args.focus}")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Thinking Mode: {args.thinking_mode}")
            print(f"Continuation ID: {thread['thread_id']}")
            print("Confidence: medium (requires validation)")
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
