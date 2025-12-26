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
PAL Tracer - Code tracing and dependency analysis tool.

Run with uv (recommended):
    uv run scripts/pal_tracer.py --files FILE [FILE ...] --target TARGET [options]

Usage:
    python pal_tracer.py --files FILE [FILE ...] --target TARGET [options]

Options:
    --files FILE [FILE ...]    Files to analyze (required)
    --target TARGET            Function/class/method to trace (required)
    --mode MODE                Tracing mode: precision (call flow) or dependencies (structural)
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --thinking-mode MODE       Thinking budget (default: high)
    --json                     Output as JSON

Examples:
    # Trace call flow for a method
    python pal_tracer.py --files src/ --target "UserService.login" --mode precision

    # Analyze structural dependencies
    python pal_tracer.py --files src/auth.py src/user.py --target "AuthHandler" --mode dependencies

    # Continue previous analysis
    python pal_tracer.py --files src/ --target "Database.query" --continuation-id abc123
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402
from conversation import ConversationMemory
from file_utils import read_files

from config import load_config
from providers import execute_request, get_provider


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser(description="PAL Tracer - Code tracing and dependency analysis")
    parser.add_argument("--files", nargs="+", required=True, help="Files to analyze")
    parser.add_argument(
        "--target",
        required=True,
        help="Function/class/method to trace (e.g., 'UserService.login')",
    )
    parser.add_argument(
        "--mode",
        choices=["precision", "dependencies"],
        default="precision",
        help="Tracing mode: precision (call flow) or dependencies (structural)",
    )
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking budget (default: high)",
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
                thread = memory.create_thread("tracer", {"files": args.files, "target": args.target})
        else:
            thread = memory.create_thread("tracer", {"files": args.files, "target": args.target})

        # Load system prompt
        system_prompt = load_prompt("tracer")

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

        user_prompt_parts.append("\n=== TRACE TARGET ===")
        user_prompt_parts.append(f"Target: {args.target}")
        user_prompt_parts.append(f"Mode: {args.mode}")

        # Add mode-specific instructions
        if args.mode == "precision":
            user_prompt_parts.append("\n=== TRACING INSTRUCTIONS ===")
            user_prompt_parts.append(
                "Perform PRECISION tracing (execution flow analysis):\n"
                "1. Locate the target function/class/method in the codebase\n"
                "2. Trace what the target calls (outgoing calls)\n"
                "3. Trace what calls the target (incoming calls)\n"
                "4. Map the data flow through the execution path\n"
                "5. Identify any entry points that trigger this code\n"
                "6. Note side effects (database, network, filesystem, state changes)"
            )
        else:  # dependencies mode
            user_prompt_parts.append("\n=== DEPENDENCY ANALYSIS INSTRUCTIONS ===")
            user_prompt_parts.append(
                "Perform DEPENDENCIES analysis (structural mapping):\n"
                "1. Locate the target in the codebase\n"
                "2. Map incoming dependencies (what modules/classes use the target)\n"
                "3. Map outgoing dependencies (what the target imports/uses)\n"
                "4. Identify type relationships (implements, extends, uses)\n"
                "5. Note coupling patterns and dependency direction\n"
                "6. Identify potential circular dependencies"
            )

        user_prompt_parts.append("\n=== OUTPUT REQUEST ===")
        user_prompt_parts.append(
            "Provide the analysis following the output format in your system prompt. "
            "Include file paths and line numbers for all references. "
            "Summarize key findings at the end."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Execute request with high thinking mode by default
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
            content=f"Trace {args.target} in files: {', '.join(args.files)} (mode: {args.mode})",
            files=args.files,
            tool_name="tracer",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="tracer",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build agentic result with confidence="medium" for complex analysis
        result = build_agentic_response(
            tool_name="tracer",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence="medium",
            files_examined=args.files,
            findings_summary=f"Traced {args.target} using {args.mode} mode",
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output with agentic metadata
            print(f"\n{'=' * 60}")
            print("Code Tracer")
            print(f"Target: {args.target}")
            print(f"Mode: {args.mode}")
            print(f"Files: {', '.join(args.files)}")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Continuation ID: {thread['thread_id']}")
            print("Confidence: medium")
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
