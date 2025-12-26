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
PAL Chat - Multi-turn conversational chat with external AI models.

Run with uv (recommended):
    uv run scripts/pal_chat.py --prompt "Your question" [options]

Usage:
    python pal_chat.py --prompt "Your question" [options]

Options:
    --files FILE [FILE ...]      Files to include as context
    --images IMAGE [IMAGE ...]   Images to include (paths or data URLs)
    --model MODEL                Model to use (default: from config or auto)
    --continuation-id ID         Continue existing conversation
    --working-dir PATH           Working directory for artifacts
    --temperature FLOAT          Temperature (0.0-2.0)
    --thinking-mode MODE         Thinking mode (minimal/low/medium/high/max)

Examples:
    # Simple question
    python pal_chat.py --prompt "How do I implement a binary search?"

    # With file context
    python pal_chat.py --prompt "Review this code" --files src/main.py

    # Continue conversation
    python pal_chat.py --prompt "What about edge cases?" --continuation-id <uuid>

    # Use specific model
    python pal_chat.py --prompt "Explain this" --model gemini-2.5-pro
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
    parser = argparse.ArgumentParser(description="PAL Chat - Multi-turn conversational chat with external AI models")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--files", nargs="*", default=[], help="Files to include as context")
    parser.add_argument("--images", nargs="*", default=[], help="Images to include")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument("--working-dir", help="Working directory for artifacts")
    parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        help="Thinking mode for extended reasoning",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()

        # Initialize conversation memory
        memory = ConversationMemory(config)

        # Get model first for token budget calculation
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Get or create conversation thread
        conversation_history = ""
        history_tokens = 0
        if args.continuation_id:
            thread = memory.get_thread(args.continuation_id)
            if thread:
                # Use token-aware history building
                conversation_history, history_tokens = memory.build_history_with_budget(thread, resolved_model)
            else:
                # Thread not found, create new one
                thread = memory.create_thread("chat", {"prompt": args.prompt})
        else:
            thread = memory.create_thread("chat", {"prompt": args.prompt})

        # Load system prompt
        system_prompt = load_prompt("chat")

        # Read files if provided
        file_content = ""
        if args.files:
            file_content = read_files(args.files, include_line_numbers=True)

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        if file_content:
            user_prompt_parts.append("=== FILES ===")
            user_prompt_parts.append(file_content)

        user_prompt_parts.append("=== USER REQUEST ===")
        user_prompt_parts.append(args.prompt)

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get temperature and thinking mode
        temperature = args.temperature
        if temperature is None:
            temperature = config.get("defaults", {}).get("temperature", 1.0)

        thinking_mode = args.thinking_mode
        if thinking_mode is None:
            thinking_mode = config.get("defaults", {}).get("thinking_mode", "medium")

        # Execute request
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            thinking_mode=thinking_mode,
            images=args.images if args.images else None,
            config=config,
        )

        # Record turns for continuation
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=args.prompt,
            files=args.files if args.files else None,
            images=args.images if args.images else None,
            tool_name="chat",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="chat",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Infer confidence from response
        confidence = infer_confidence(
            response["content"],
            has_errors=False,
            has_warnings=False,
            has_actionable_items=True,
        )

        # Build agentic result with rich metadata
        result = build_agentic_response(
            tool_name="chat",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence=confidence,
            files_examined=args.files if args.files else [],
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\n{'=' * 60}")
            print(f"Model: {resolved_model} via {provider.provider_name}")
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
