#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL API Lookup - Returns orchestration instructions for API documentation lookup.

This tool does NOT make AI model calls. Instead, it provides search strategy
instructions for Claude to use WebSearch tool to find current documentation.

Run with uv (recommended):
    uv run scripts/pal_apilookup.py --prompt "API/SDK to look up" [options]

Usage:
    python pal_apilookup.py --prompt "API/SDK to look up" [options]

Options:
    --json               Output as JSON

Examples:
    # Look up React hooks documentation
    python pal_apilookup.py --prompt "React useEffect hook"

    # Look up Python library
    python pal_apilookup.py --prompt "pandas DataFrame merge"

    # JSON output for programmatic use
    python pal_apilookup.py --prompt "FastAPI dependency injection" --json
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402


def build_search_instructions(query: str) -> str:
    """Build search strategy instructions for the API lookup."""
    return f"""## API Documentation Lookup Instructions

**Query**: {query}

### Search Strategy

Use WebSearch tool to find current documentation. Follow this process:

1. **Primary Search** (required): Search for official documentation
   - Query: "{query} official documentation"
   - Prioritize results from official domains (.io, .dev, .org, github.com)

2. **Version Check** (if needed): If user needs specific version
   - Query: "{query} version [X.Y] documentation"
   - Or check latest version info first

3. **API Reference** (if needed): For specific method/function details
   - Query: "{query} API reference"
   - Look for type signatures, parameters, return values

4. **Examples** (if needed): For usage patterns
   - Query: "{query} examples tutorial"
   - Prioritize official examples over third-party

### Rules

- **STOP after 2-4 searches maximum** - do not over-search
- Prioritize official sources over blog posts
- Check documentation dates/versions for currency
- If official docs not found in first 2 searches, report to user

### Source Priority

1. Official documentation sites
2. GitHub repositories (README, docs folder)
3. Stack Overflow (for edge cases)
4. Reputable technical blogs (only if official docs lacking)"""


def main():
    parser = argparse.ArgumentParser(
        description="PAL API Lookup - Returns orchestration instructions for API documentation lookup"
    )
    parser.add_argument("--prompt", required=True, help="API/SDK/library to look up")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Generate a session ID for tracking
        session_id = str(uuid.uuid4())

        # Build search instructions
        instructions = build_search_instructions(args.prompt)

        # Build agentic response with needs_escalation status
        result = build_agentic_response(
            tool_name="apilookup",
            status="needs_escalation",
            content=instructions,
            continuation_id=session_id,
            confidence="medium",
            additional_next_actions=[
                f"Use WebSearch to find documentation for: {args.prompt}",
                "Follow the search strategy above (2-4 searches max)",
                "Summarize findings with version info and key points",
            ],
        )

        # Add custom fields for orchestration
        result["web_lookup_needed"] = True
        result["user_prompt"] = args.prompt

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\n{'=' * 60}")
            print(f"API Lookup Request: {args.prompt}")
            print(f"Session ID: {session_id}")
            print(f"{'=' * 60}\n")
            print(instructions)
            print(f"\n{'=' * 60}")
            print("Status: needs_escalation (Claude should use WebSearch)")
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
