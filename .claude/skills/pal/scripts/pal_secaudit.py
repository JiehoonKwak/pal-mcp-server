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
PAL SecAudit - Security audit tool with OWASP Top 10 analysis.

Run with uv (recommended):
    uv run scripts/pal_secaudit.py --files FILE [FILE ...] [options]

Usage:
    python pal_secaudit.py --files FILE [FILE ...] [options]

Options:
    --files FILE [FILE ...]    Files to audit (required)
    --focus AREA               Focus area (owasp, compliance, infrastructure, dependencies, comprehensive)
    --threat-level LEVEL       Threat level: low, medium, high, critical (default: medium)
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --thinking-mode MODE       Thinking mode (default: high)
    --json                     Output as JSON

Examples:
    # Basic security audit
    python pal_secaudit.py --files src/auth.py

    # Audit with specific focus
    python pal_secaudit.py --files src/*.py --focus owasp --threat-level high

    # Comprehensive audit
    python pal_secaudit.py --files src/ --focus comprehensive
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402
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


def infer_security_confidence(response_content: str) -> str:
    """
    Infer confidence based on security severity levels found in response.

    Returns 'high' for security audits since they require certainty.
    Downgrades to 'medium' if response contains uncertainty indicators.
    """
    content_lower = response_content.lower()

    # Check for critical/high severity findings - these need high confidence
    has_critical = bool(re.search(r"\[critical\]", content_lower))
    has_high = bool(re.search(r"\[high\]", content_lower))

    # Check for uncertainty indicators
    uncertainty_phrases = [
        "might be vulnerable",
        "could potentially",
        "needs further investigation",
        "unclear whether",
        "may require additional",
    ]
    has_uncertainty = any(phrase in content_lower for phrase in uncertainty_phrases)

    if has_uncertainty:
        return "medium"

    # Security audits with critical/high findings need high confidence
    if has_critical or has_high:
        return "high"

    return "high"


def main():
    parser = argparse.ArgumentParser(description="PAL SecAudit - Security audit with OWASP Top 10 analysis")
    parser.add_argument("--files", nargs="+", required=True, help="Files to audit")
    parser.add_argument(
        "--focus",
        choices=["owasp", "compliance", "infrastructure", "dependencies", "comprehensive"],
        default="owasp",
        help="Focus area for the audit",
    )
    parser.add_argument(
        "--threat-level",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Threat level context (default: medium)",
    )
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing conversation")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking mode (default: high)",
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
                thread = memory.create_thread("secaudit", {"files": args.files})
        else:
            thread = memory.create_thread("secaudit", {"files": args.files})

        # Load system prompt
        system_prompt = load_prompt("secaudit")

        # Read files with line numbers for precise vulnerability reporting
        file_content = read_files(args.files, include_line_numbers=True)

        if not file_content.strip():
            raise ValueError("No valid files found to audit")

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        user_prompt_parts.append("=== CODE TO AUDIT ===")
        user_prompt_parts.append(file_content)

        # Add focus area context
        focus_instructions = {
            "owasp": "Focus on OWASP Top 10 (2021) vulnerabilities.",
            "compliance": "Focus on compliance issues (GDPR, HIPAA, PCI-DSS, SOC2).",
            "infrastructure": "Focus on infrastructure security (secrets, configs, IAM).",
            "dependencies": "Focus on dependency vulnerabilities and supply chain risks.",
            "comprehensive": "Perform comprehensive security audit covering all areas.",
        }

        user_prompt_parts.append("\n=== AUDIT CONTEXT ===")
        user_prompt_parts.append(f"Focus: {focus_instructions[args.focus]}")
        user_prompt_parts.append(f"Threat Level: {args.threat_level}")

        if args.threat_level in ("high", "critical"):
            user_prompt_parts.append(
                "This is a high-priority security review. Be thorough and flag any potential issues."
            )

        user_prompt_parts.append("\n=== AUDIT REQUEST ===")
        user_prompt_parts.append(
            "Perform a security audit on the code above following your system prompt guidelines. "
            "Report all vulnerabilities with severity levels (CRITICAL, HIGH, MEDIUM, LOW), "
            "specific file locations with line numbers, and actionable remediation steps."
        )

        full_user_prompt = "\n\n".join(user_prompt_parts)

        # Get model and provider
        model = args.model or config.get("defaults", {}).get("model", "auto")
        provider, resolved_model = get_provider(model, config)

        # Execute request with high thinking mode for thorough analysis
        response = execute_request(
            provider=provider,
            prompt=full_user_prompt,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=0.3,  # Low temperature for precise security analysis
            thinking_mode=args.thinking_mode,
            config=config,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Security audit: {', '.join(args.files)}, focus={args.focus}, threat-level={args.threat_level}",
            files=args.files,
            tool_name="secaudit",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="secaudit",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Infer confidence from response severity analysis
        confidence = infer_security_confidence(response["content"])

        # Build agentic result
        result = build_agentic_response(
            tool_name="secaudit",
            status="success",
            content=response["content"],
            continuation_id=thread["thread_id"],
            model=resolved_model,
            provider=provider.provider_name,
            usage=response.get("usage", {}),
            confidence=confidence,
            files_examined=args.files,
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print(f"\n{'=' * 60}")
            print("Security Audit")
            print(f"Files: {', '.join(args.files)}")
            print(f"Focus: {args.focus}")
            print(f"Threat Level: {args.threat_level}")
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
