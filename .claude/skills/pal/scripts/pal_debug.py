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
PAL Debug - Expert debugging analysis with external AI models.

Provides systematic debugging assistance for complex issues with
hypothesis generation, root cause analysis, and fix recommendations.

Run with uv (recommended):
    uv run scripts/pal_debug.py --issue "Bug description" --files FILE [options]

Usage:
    python pal_debug.py --issue "Bug description" --files FILE [options]

Options:
    --issue ISSUE            Description of the bug/issue (required)
    --files FILE [...]       Files to analyze (required)
    --error-logs LOGS        Error logs or stack traces
    --model MODEL            Model to use (default: from config)
    --continuation-id ID     Continue existing debugging session

Examples:
    # Debug with files
    python pal_debug.py --issue "API returns 500 on login" \\
        --files src/auth.py src/api.py

    # Debug with error logs
    python pal_debug.py --issue "Memory leak in worker" \\
        --files src/worker.py \\
        --error-logs "OOMKilled at 2GB after 4 hours"

    # Continue debugging session
    python pal_debug.py --issue "Check hypothesis 2" \\
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
        description="PAL Debug - Expert debugging analysis with external AI models"
    )
    parser.add_argument("--issue", required=True, help="Description of the bug/issue")
    parser.add_argument("--files", nargs="+", required=True, help="Files to analyze")
    parser.add_argument("--error-logs", help="Error logs or stack traces")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing debugging session")
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
                thread = memory.create_thread("debug", {"issue": args.issue})
        else:
            thread = memory.create_thread("debug", {"issue": args.issue})

        # Load system prompt
        system_prompt = load_prompt("debug")

        # Read files
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to analyze")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== ISSUE DESCRIPTION ===")
        user_prompt_parts.append(args.issue)

        if args.error_logs:
            user_prompt_parts.append("\n=== ERROR LOGS ===")
            user_prompt_parts.append(args.error_logs)

        user_prompt_parts.append("\n=== CODE FILES ===")
        user_prompt_parts.append(file_content)

        user_prompt_parts.append("\n=== DEBUGGING REQUEST ===")
        user_prompt_parts.append(
            "Please analyze the issue above using systematic debugging methodology. "
            "Generate hypotheses ranked by likelihood, identify root causes, "
            "and provide minimal fixes with regression prevention."
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
            temperature=0.5,  # Lower temperature for precise analysis
            thinking_mode=args.thinking_mode,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Debug: {args.issue}",
            files=args.files,
            tool_name="debug",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="debug",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build result
        result = {
            "status": "success",
            "content": response["content"],
            "continuation_id": thread["thread_id"],
            "issue": args.issue,
            "files_analyzed": args.files,
            "model": resolved_model,
            "provider": provider.provider_name,
            "usage": response.get("usage", {}),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"Debug Analysis")
            print(f"Issue: {args.issue}")
            print(f"Files: {', '.join(args.files)}")
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
