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
PAL DocGen - Documentation generation with external AI models.

Run with uv (recommended):
    uv run scripts/pal_docgen.py --files FILE [FILE ...] [options]

Usage:
    python pal_docgen.py --files FILE [FILE ...] [options]

Options:
    --files FILE [FILE ...]    Files to document (required)
    --format FORMAT            Output format (markdown, docstring, jsdoc)
    --include-complexity       Include Big O analysis (default: true)
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --json                     Output as JSON

Examples:
    # Generate docstrings for Python files
    python pal_docgen.py --files src/main.py

    # Generate JSDoc for JavaScript
    python pal_docgen.py --files src/utils.js --format jsdoc

    # Generate markdown documentation
    python pal_docgen.py --files src/*.py --format markdown
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
    parser = argparse.ArgumentParser(description="PAL DocGen - Documentation generation with external AI models")
    parser.add_argument("--files", nargs="+", required=True, help="Files to document")
    parser.add_argument(
        "--format",
        choices=["markdown", "docstring", "jsdoc"],
        default="docstring",
        help="Output format (default: docstring)",
    )
    parser.add_argument(
        "--include-complexity",
        action="store_true",
        default=True,
        help="Include Big O analysis (default: true)",
    )
    parser.add_argument(
        "--no-complexity",
        action="store_true",
        help="Exclude Big O analysis",
    )
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Handle --no-complexity flag
    include_complexity = not args.no_complexity

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
                thread = memory.create_thread("docgen", {"files": args.files})
        else:
            thread = memory.create_thread("docgen", {"files": args.files})

        # Load system prompt
        system_prompt = load_prompt("docgen")

        # Read files
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to document")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== CODE TO DOCUMENT ===")
        user_prompt_parts.append(file_content)

        # Add format instructions
        user_prompt_parts.append("\n=== OUTPUT FORMAT ===")
        format_instructions = {
            "markdown": "Generate comprehensive markdown documentation suitable for README or API docs.",
            "docstring": "Add inline docstrings/comments using language-appropriate conventions (Google-style for Python, JSDoc for JS/TS).",
            "jsdoc": "Generate JSDoc-style documentation blocks.",
        }
        user_prompt_parts.append(format_instructions[args.format])

        # Add complexity analysis instruction
        if include_complexity:
            user_prompt_parts.append("\n=== COMPLEXITY ANALYSIS ===")
            user_prompt_parts.append(
                "Include Big O time and space complexity analysis for each function/method. "
                "Explain the reasoning behind each complexity assessment."
            )

        user_prompt_parts.append("\n=== DOCUMENTATION REQUEST ===")
        user_prompt_parts.append(
            "Please generate comprehensive documentation for the code above following the guidelines in your system prompt. "
            "Ensure all parameters, return values, and exceptions are documented clearly."
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
            temperature=0.5,  # Lower temperature for consistent documentation
            config=config,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Document files: {', '.join(args.files)} (format: {args.format})",
            files=args.files,
            tool_name="docgen",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="docgen",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build agentic result with high confidence (documentation is deterministic)
        result = build_agentic_response(
            tool_name="docgen",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence="high",
            files_examined=args.files,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output with agentic metadata
            print(f"\n{'=' * 60}")
            print("Documentation Generated")
            print(f"Files: {', '.join(args.files)}")
            print(f"Format: {args.format}")
            print(f"Complexity Analysis: {'included' if include_complexity else 'excluded'}")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Continuation ID: {thread['thread_id']}")
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
