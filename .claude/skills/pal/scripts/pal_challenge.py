#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL Challenge - Wrap statements in critical thinking instructions.

This tool does NOT make AI model calls. It transforms input by wrapping
statements/claims in challenge instructions for critical reassessment.

Run with uv (recommended):
    uv run scripts/pal_challenge.py --prompt "Your statement" [options]

Usage:
    python pal_challenge.py --prompt "Your statement" [options]

Options:
    --json    Output as JSON

Examples:
    # Challenge a statement
    python pal_challenge.py --prompt "This approach is optimal"

    # JSON output
    python pal_challenge.py --prompt "We should use microservices" --json
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402

CHALLENGE_TEMPLATE = """CRITICAL REASSESSMENT – Do not automatically agree:

"{prompt}"

Carefully evaluate the statement above. Is it accurate, complete, and well-reasoned?
Investigate if needed before replying, and stay focused. If you identify flaws, gaps, or misleading
points, explain them clearly. Likewise, if you find the reasoning sound, explain why it holds up.
Respond with thoughtful analysis—stay to the point and avoid reflexive agreement."""


def build_challenge_prompt(statement: str) -> str:
    """Wrap a statement in critical thinking instructions."""
    return CHALLENGE_TEMPLATE.format(prompt=statement)


def main():
    parser = argparse.ArgumentParser(description="PAL Challenge - Wrap statements in critical thinking instructions")
    parser.add_argument("--prompt", required=True, help="Statement to challenge")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Build the challenge prompt (no AI call needed)
        challenge_prompt = build_challenge_prompt(args.prompt)

        # Generate a unique ID for tracking
        challenge_id = str(uuid.uuid4())

        # Build response content
        content = f"""## Challenge Prompt Generated

**Original Statement:**
{args.prompt}

**Challenge Prompt:**
{challenge_prompt}

---

**Instructions for Claude:**
1. Process the challenge prompt above with thorough critical analysis
2. Do not reflexively agree - evaluate the statement objectively
3. Investigate relevant context if needed before responding
4. Clearly explain any flaws, gaps, or strengths identified
5. Provide a well-reasoned conclusion"""

        # Build agentic result (next_actions come from TOOL_CONFIGS in agentic.py)
        result = build_agentic_response(
            tool_name="challenge",
            status="success",
            content=content,
            continuation_id=challenge_id,
            confidence="high",
        )

        # Add challenge-specific fields to the result
        result["original_statement"] = args.prompt
        result["challenge_prompt"] = challenge_prompt

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\n{'=' * 60}")
            print("PAL Challenge - Critical Reassessment")
            print(f"Challenge ID: {challenge_id}")
            print(f"{'=' * 60}\n")
            print(f"Original Statement:\n  {args.prompt}\n")
            print(f"Challenge Prompt:\n{challenge_prompt}")
            print(f"\n{'=' * 60}")
            print("Instructions: Process this with chat or thinkdeep for analysis")
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
