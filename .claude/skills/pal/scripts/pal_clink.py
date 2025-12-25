#!/usr/bin/env python3
"""
PAL Clink - CLI-to-CLI bridge for spawning AI CLIs.

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
import subprocess
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from config import load_cli_client, load_config


def load_role_prompt(role_path: str) -> str:
    """Load a role prompt file."""
    skill_root = Path(__file__).parent.parent
    prompt_path = skill_root / role_path

    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


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
        # Load configuration
        config = load_config()

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

        full_prompt = "\n\n".join(full_prompt_parts)

        # Build command
        cmd = [client["command"]]
        cmd.extend(client.get("additional_args", []))
        cmd.extend(role.get("role_args", []))

        # Add files if provided
        for file_path in args.files:
            # Different CLIs have different file argument formats
            if args.cli == "claude":
                cmd.extend(["--add-file", file_path])
            elif args.cli == "gemini":
                cmd.extend(["--file", file_path])
            else:
                cmd.extend(["--file", file_path])

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

            output = {
                "status": "success" if result.returncode == 0 else "error",
                "content": result.stdout,
                "stderr": result.stderr if result.stderr else None,
                "return_code": result.returncode,
                "cli": args.cli,
                "role": args.role,
            }

        except subprocess.TimeoutExpired:
            output = {
                "status": "error",
                "error": f"CLI '{args.cli}' timed out after {args.timeout} seconds",
                "cli": args.cli,
                "role": args.role,
            }

        except FileNotFoundError:
            output = {
                "status": "error",
                "error": f"CLI '{args.cli}' not found. Is it installed and in PATH?",
                "cli": args.cli,
                "role": args.role,
            }

        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"CLI: {args.cli} | Role: {args.role}")
            print(f"Status: {output['status']}")
            print(f"{'='*60}\n")

            if output.get("content"):
                print(output["content"])

            if output.get("stderr"):
                print(f"\n--- STDERR ---\n{output['stderr']}")

            if output.get("error"):
                print(f"\nError: {output['error']}")

            print(f"\n{'='*60}\n")

        if output["status"] == "error":
            sys.exit(1)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        if args.json:
            print(json.dumps(error_result, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
