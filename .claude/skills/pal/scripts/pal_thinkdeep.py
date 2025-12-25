#!/usr/bin/env python3
"""
PAL ThinkDeep - Deep systematic analysis with extended thinking.

Usage:
    python pal_thinkdeep.py --prompt "Complex problem to analyze" [options]

Options:
    --files FILE [FILE ...]      Files to include as context
    --model MODEL                Model to use (default: from config or auto)
    --continuation-id ID         Continue existing conversation
    --thinking-mode MODE         Thinking mode (minimal/low/medium/high/max)
                                 Default: high (for deep analysis)

Examples:
    # Deep analysis of architecture
    python pal_thinkdeep.py --prompt "Analyze the trade-offs of this design"

    # With file context
    python pal_thinkdeep.py --prompt "What are the performance implications?" \\
        --files src/core.py --thinking-mode max

    # Continue deep analysis
    python pal_thinkdeep.py --prompt "Explore the security angle" \\
        --continuation-id <uuid>
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
        description="PAL ThinkDeep - Deep systematic analysis with extended thinking"
    )
    parser.add_argument("--prompt", required=True, help="Problem or question to analyze deeply")
    parser.add_argument("--files", nargs="*", default=[], help="Files to include as context")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",  # ThinkDeep defaults to high thinking
        help="Thinking mode for extended reasoning (default: high)",
    )
    parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
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
                thread = memory.create_thread("thinkdeep", {"prompt": args.prompt})
        else:
            thread = memory.create_thread("thinkdeep", {"prompt": args.prompt})

        # Load system prompt
        system_prompt = load_prompt("thinkdeep")

        # Read files if provided
        file_content = ""
        if args.files:
            file_content = read_files(args.files, include_line_numbers=True)

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        if file_content:
            user_prompt_parts.append("=== CONTEXT FILES ===")
            user_prompt_parts.append(file_content)

        user_prompt_parts.append("=== ANALYSIS REQUEST ===")
        user_prompt_parts.append(args.prompt)
        user_prompt_parts.append("")
        user_prompt_parts.append(
            "Please provide a thorough, systematic analysis. "
            "Consider multiple angles, surface potential issues, "
            "and provide actionable recommendations."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Get temperature (ThinkDeep typically uses lower temperature for precision)
        temperature = args.temperature
        if temperature is None:
            temperature = config.get("defaults", {}).get("temperature", 0.7)

        # Execute request with extended thinking
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            thinking_mode=args.thinking_mode,
        )

        # Record turns for continuation
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=args.prompt,
            files=args.files if args.files else None,
            tool_name="thinkdeep",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="thinkdeep",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build result
        result = {
            "status": "success",
            "content": response["content"],
            "continuation_id": thread["thread_id"],
            "model": resolved_model,
            "provider": provider.provider_name,
            "thinking_mode": args.thinking_mode,
            "usage": response.get("usage", {}),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\n{'='*60}")
            print(f"ThinkDeep Analysis")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Thinking Mode: {args.thinking_mode}")
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
