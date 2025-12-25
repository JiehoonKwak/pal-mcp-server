#!/usr/bin/env python3
"""
PAL CodeReview - Comprehensive code review with external AI models.

Usage:
    python pal_codereview.py --files FILE [FILE ...] [options]

Options:
    --files FILE [FILE ...]    Files to review (required)
    --prompt PROMPT            Additional review instructions
    --focus AREA [AREA ...]    Focus areas (security, performance, etc.)
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing review conversation

Examples:
    # Basic code review
    python pal_codereview.py --files src/main.py

    # Review with focus
    python pal_codereview.py --files src/auth.py --focus security

    # Review multiple files
    python pal_codereview.py --files src/*.py --focus performance maintainability
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
        description="PAL CodeReview - Comprehensive code review with external AI models"
    )
    parser.add_argument("--files", nargs="+", required=True, help="Files to review")
    parser.add_argument("--prompt", help="Additional review instructions")
    parser.add_argument(
        "--focus",
        nargs="*",
        default=[],
        choices=["security", "performance", "maintainability", "architecture", "testing"],
        help="Focus areas for the review",
    )
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument("--thinking-mode", choices=["minimal", "low", "medium", "high", "max"])
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
                thread = memory.create_thread("codereview", {"files": args.files})
        else:
            thread = memory.create_thread("codereview", {"files": args.files})

        # Load system prompt
        system_prompt = load_prompt("codereview")

        # Read files
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to review")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== CODE TO REVIEW ===")
        user_prompt_parts.append(file_content)

        # Add focus areas
        if args.focus:
            user_prompt_parts.append(f"\n=== FOCUS AREAS ===")
            user_prompt_parts.append(f"Please focus on: {', '.join(args.focus)}")

        # Add custom prompt
        if args.prompt:
            user_prompt_parts.append(f"\n=== ADDITIONAL INSTRUCTIONS ===")
            user_prompt_parts.append(args.prompt)

        user_prompt_parts.append("\n=== REVIEW REQUEST ===")
        user_prompt_parts.append(
            "Please review the code above following the guidelines in your system prompt. "
            "Identify issues by severity (Critical > High > Medium > Low) and provide "
            "actionable fixes with specific line references."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Get thinking mode
        thinking_mode = args.thinking_mode
        if thinking_mode is None:
            thinking_mode = config.get("defaults", {}).get("thinking_mode", "medium")

        # Execute request
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=0.7,  # Lower temperature for precise analysis
            thinking_mode=thinking_mode,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Review files: {', '.join(args.files)}" + (f"\n{args.prompt}" if args.prompt else ""),
            files=args.files,
            tool_name="codereview",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="codereview",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build result
        result = {
            "status": "success",
            "content": response["content"],
            "continuation_id": thread["thread_id"],
            "files_reviewed": args.files,
            "focus_areas": args.focus if args.focus else None,
            "model": resolved_model,
            "provider": provider.provider_name,
            "usage": response.get("usage", {}),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Code Review")
            print(f"Files: {', '.join(args.files)}")
            if args.focus:
                print(f"Focus: {', '.join(args.focus)}")
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
