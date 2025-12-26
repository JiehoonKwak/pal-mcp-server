#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///
"""
PAL Clink - CLI-to-CLI bridge for spawning AI CLIs.

Run with uv (recommended):
    uv run scripts/pal_clink.py --cli gemini --prompt "Your request" [options]

Usage:
    python pal_clink.py --cli <cli_name> --prompt "Your request" [options]

Options:
    --cli CLI              CLI to use (gemini, claude, codex)
    --role ROLE            Role preset (default, planner, codereviewer)
    --prompt PROMPT        Prompt to send
    --files FILE [...]     Files to include
    --timeout SECONDS      Timeout in seconds (default: 300)

Examples:
    # Use Gemini CLI with default role
    python pal_clink.py --cli gemini --prompt "Analyze this codebase"

    # Use Claude as a planner
    python pal_clink.py --cli claude --role planner --prompt "Plan feature X"

    # Use Gemini as code reviewer with files
    python pal_clink.py --cli gemini --role codereviewer \\
        --prompt "Review this code" --files src/main.py
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402

from config import load_cli_client

# Output size limits (matches MCP server)
MAX_RESPONSE_CHARS = 20_000
SUMMARY_PATTERN = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.IGNORECASE | re.DOTALL)


def load_role_prompt(role_path: str) -> str:
    """Load a role prompt file."""
    skill_root = Path(__file__).parent.parent
    prompt_path = skill_root / role_path

    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def extract_summary(content: str) -> str | None:
    """Extract <SUMMARY> block from content if present."""
    match = SUMMARY_PATTERN.search(content)
    if not match:
        return None
    summary = match.group(1).strip()
    return summary or None


def format_file_references(files: list[str]) -> str:
    """Format file references with metadata (size, modified time)."""
    if not files:
        return ""

    references = []
    for file_path in files:
        try:
            path = Path(file_path)
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            size = stat.st_size
            references.append(f"- {file_path} (last modified {modified}, {size} bytes)")
        except OSError:
            references.append(f"- {file_path} (unavailable)")
    return "\n".join(references)


def apply_output_limit(content: str, cli_name: str) -> tuple[str, dict]:
    """
    Apply output size limit with summary extraction.

    Returns:
        Tuple of (processed_content, metadata_dict)
    """
    metadata = {}

    if len(content) <= MAX_RESPONSE_CHARS:
        return content, metadata

    # Try to extract summary
    summary = extract_summary(content)
    if summary:
        if len(summary) <= MAX_RESPONSE_CHARS:
            metadata.update(
                {
                    "output_summarized": True,
                    "output_original_length": len(content),
                    "output_summary_length": len(summary),
                    "output_limit": MAX_RESPONSE_CHARS,
                }
            )
            return summary, metadata
        # Summary too long, truncate it
        summary = summary[:MAX_RESPONSE_CHARS]
        metadata.update(
            {
                "output_summarized": True,
                "output_truncated": True,
                "output_original_length": len(content),
                "output_limit": MAX_RESPONSE_CHARS,
            }
        )
        return summary, metadata

    # No summary, truncate with excerpt
    excerpt_limit = min(4000, MAX_RESPONSE_CHARS // 2)
    excerpt = content[:excerpt_limit]
    metadata.update(
        {
            "output_truncated": True,
            "output_original_length": len(content),
            "output_excerpt_length": len(excerpt),
            "output_limit": MAX_RESPONSE_CHARS,
        }
    )

    message = (
        f"CLI '{cli_name}' produced {len(content)} characters, exceeding the limit "
        f"({MAX_RESPONSE_CHARS} characters). Please narrow the request or run the CLI directly.\n\n"
        f"--- Begin excerpt ({len(excerpt)} of {len(content)} chars) ---\n{excerpt}\n--- End excerpt ---"
    )
    return message, metadata


def main():
    parser = argparse.ArgumentParser(description="PAL Clink - CLI-to-CLI bridge")
    parser.add_argument(
        "--cli",
        required=True,
        choices=["gemini", "claude", "codex"],
        help="CLI to use",
    )
    parser.add_argument(
        "--role",
        default="default",
        help="Role preset (default, planner, codereviewer)",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to send")
    parser.add_argument("--files", nargs="*", default=[], help="Files to include")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Load CLI client configuration
        try:
            client = load_cli_client(args.cli)
        except FileNotFoundError:
            # Create default client config
            client = {
                "name": args.cli,
                "command": args.cli,
                "additional_args": [],
                "env": {},
                "roles": {
                    "default": {"prompt_path": "prompts/clink/default.md", "role_args": []},
                    "planner": {"prompt_path": "prompts/clink/planner.md", "role_args": []},
                    "codereviewer": {
                        "prompt_path": "prompts/clink/codereviewer.md",
                        "role_args": [],
                    },
                },
            }

        # Get role configuration
        role = client.get("roles", {}).get(args.role)
        if not role:
            role = client.get("roles", {}).get("default", {})

        # Load role prompt
        role_prompt = ""
        if role.get("prompt_path"):
            role_prompt = load_role_prompt(role["prompt_path"])

        # Build full prompt
        full_prompt_parts = []

        if role_prompt:
            full_prompt_parts.append(role_prompt)

        full_prompt_parts.append("=== USER REQUEST ===")
        full_prompt_parts.append(args.prompt)

        # Add file references with metadata
        if args.files:
            file_refs = format_file_references(args.files)
            if file_refs:
                full_prompt_parts.append("=== FILE REFERENCES ===")
                full_prompt_parts.append(file_refs)

        full_prompt = "\n\n".join(full_prompt_parts)

        # Build command
        cmd = [client["command"]]
        cmd.extend(client.get("additional_args", []))
        cmd.extend(role.get("role_args", []))

        # NOTE: Files are embedded as references in the prompt (format_file_references)
        # matching the original MCP clink architecture. CLIs don't receive --file args.

        # Add prompt (different CLIs have different prompt formats)
        if args.cli == "claude":
            cmd.extend(["--prompt", full_prompt])
        elif args.cli == "gemini":
            cmd.extend(["-p", full_prompt])
        else:
            cmd.extend(["--prompt", full_prompt])

        # Build environment
        env = dict(os.environ)
        env.update(client.get("env", {}))

        # Execute CLI
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=env,
            )

            # Apply output limit with summary extraction
            content = result.stdout
            limit_metadata = {}
            if content:
                content, limit_metadata = apply_output_limit(content, args.cli)

            # Build agentic result
            output = build_agentic_response(
                tool_name="clink",
                status="success" if result.returncode == 0 else "error",
                content=content,
                continuation_id="",  # Clink doesn't use continuation
                model=f"CLI: {args.cli}",
                provider=args.cli,
                confidence="medium",  # CLI responses have medium confidence
            )
            output["cli"] = args.cli
            output["role"] = args.role
            output["return_code"] = result.returncode
            if result.stderr:
                output["stderr"] = result.stderr

            # Add limit metadata if output was modified
            if limit_metadata:
                output.update(limit_metadata)

        except subprocess.TimeoutExpired:
            output = build_agentic_response(
                tool_name="clink",
                status="error",
                content=f"CLI '{args.cli}' timed out after {args.timeout} seconds",
                continuation_id="",
                model=f"CLI: {args.cli}",
                provider=args.cli,
                confidence="certain",  # Timeout is a certain error
            )
            output["cli"] = args.cli
            output["role"] = args.role
            output["error"] = f"CLI '{args.cli}' timed out after {args.timeout} seconds"

        except FileNotFoundError:
            output = build_agentic_response(
                tool_name="clink",
                status="error",
                content=f"CLI '{args.cli}' not found. Is it installed and in PATH?",
                continuation_id="",
                model=f"CLI: {args.cli}",
                provider=args.cli,
                confidence="certain",  # File not found is a certain error
            )
            output["cli"] = args.cli
            output["role"] = args.role
            output["error"] = f"CLI '{args.cli}' not found. Is it installed and in PATH?"

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print(f"CLI: {args.cli} | Role: {args.role}")
            print(f"Status: {output['status']}")
            print(f"Model: {output.get('model', 'N/A')}")
            print(f"Provider: {output.get('provider', 'N/A')}")
            if output.get("agentic"):
                agentic = output["agentic"]
                print(f"Confidence: {agentic.get('confidence', 'N/A')}")
                if agentic.get("escalation_path"):
                    print(f"Escalation: {agentic['escalation_path']}")
            print(f"{'=' * 60}\n")

            if output.get("content"):
                print(output["content"])

            if output.get("stderr"):
                print(f"\n--- STDERR ---\n{output['stderr']}")

            if output.get("error"):
                print(f"\nError: {output['error']}")

            # Show agentic next actions
            if output.get("agentic", {}).get("next_actions"):
                print("\n--- SUGGESTED NEXT ACTIONS ---")
                for action in output["agentic"]["next_actions"]:
                    print(f"  - {action}")

            # Show related tools
            if output.get("agentic", {}).get("related_tools"):
                print(f"\nRelated tools: {', '.join(output['agentic']['related_tools'])}")

            print(f"\n{'=' * 60}\n")

        if output["status"] == "error":
            sys.exit(1)

    except Exception as e:
        error_result = build_agentic_response(
            tool_name="clink",
            status="error",
            content=str(e),
            continuation_id="",
            model="",
            provider="",
            confidence="low",
        )
        error_result["error"] = str(e)
        error_result["error_type"] = type(e).__name__
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
