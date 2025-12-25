#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai>=1.0.0",
#     "openai>=1.0.0",
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL Analyze - Holistic technical audit and strategic analysis.

Provides high-level architectural analysis focusing on scalability,
maintainability, and strategic improvement areas.

Run with uv (recommended):
    uv run scripts/pal_analyze.py --prompt "Analysis question" --files FILE [options]

Usage:
    python pal_analyze.py --prompt "Analysis question" --files FILE [options]

Options:
    --prompt PROMPT          Analysis question or focus area (required)
    --files FILE [...]       Files/directories to analyze
    --model MODEL            Model to use (default: from config)
    --continuation-id ID     Continue existing analysis session

Examples:
    # Analyze architecture
    python pal_analyze.py --prompt "Is this architecture scalable?" \\
        --files src/

    # Analyze specific concerns
    python pal_analyze.py --prompt "Assess tech debt in the auth module" \\
        --files src/auth/ src/models/user.py

    # Strategic analysis
    python pal_analyze.py --prompt "Should we migrate to microservices?" \\
        --files src/ docs/architecture.md
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from config import load_config
from conversation import ConversationMemory
from file_utils import read_files
from providers import execute_request, get_provider


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="PAL Analyze - Holistic technical audit and strategic analysis"
    )
    parser.add_argument("--prompt", required=True, help="Analysis question or focus area")
    parser.add_argument("--files", nargs="*", default=[], help="Files/directories to analyze")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing analysis session")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking mode for analysis (default: high)",
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
                thread = memory.create_thread("analyze", {"prompt": args.prompt})
        else:
            thread = memory.create_thread("analyze", {"prompt": args.prompt})

        # Load system prompt
        system_prompt = load_prompt("analyze")

        # Read files if provided
        file_content = ""
        if args.files:
            file_content = read_files(args.files, include_line_numbers=True)

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== ANALYSIS REQUEST ===")
        user_prompt_parts.append(args.prompt)

        if file_content:
            user_prompt_parts.append("\n=== CODE/PROJECT FILES ===")
            user_prompt_parts.append(file_content)

        user_prompt_parts.append("\n=== INSTRUCTIONS ===")
        user_prompt_parts.append(
            "Provide a holistic technical audit following the analysis framework. "
            "Focus on strategic insights, architectural alignment, and actionable recommendations. "
            "Avoid line-by-line code review - that's for the codereview tool."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Execute request
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=0.7,
            thinking_mode=args.thinking_mode,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=args.prompt,
            files=args.files if args.files else None,
            tool_name="analyze",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="analyze",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build result
        result = {
            "status": "success",
            "content": response["content"],
            "continuation_id": thread["thread_id"],
            "analysis_focus": args.prompt,
            "files_analyzed": args.files if args.files else None,
            "model": resolved_model,
            "provider": provider.provider_name,
            "usage": response.get("usage", {}),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Strategic Analysis")
            print(f"Focus: {args.prompt}")
            if args.files:
                print(f"Files: {len(args.files)} analyzed")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Continuation ID: {thread['thread_id']}")
            print(f"{'='*60}\n")
            print(response["content"])
            print(f"\n{'='*60}")
            if response.get("usage"):
                usage = response["usage"]
                print(
                    f"Tokens: {usage.get('input_tokens', 'N/A')} in / "
                    f"{usage.get('output_tokens', 'N/A')} out"
                )
            print(f"{'='*60}\n")

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
