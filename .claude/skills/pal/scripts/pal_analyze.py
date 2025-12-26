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
PAL Analyze - Holistic technical audit and strategic analysis.

Provides high-level architectural analysis focusing on scalability,
maintainability, and strategic improvement areas.

Run with uv (recommended):
    uv run scripts/pal_analyze.py --prompt "Analysis question" --files FILE [options]

Usage:
    python pal_analyze.py --prompt "Analysis question" --files FILE [options]

Options:
    --prompt PROMPT          Analysis question or focus area (required)
    --files FILE [...]       Files/directories to analyze
    --model MODEL            Model to use (default: from config)
    --continuation-id ID     Continue existing analysis session

Examples:
    # Analyze architecture
    python pal_analyze.py --prompt "Is this architecture scalable?" \\
        --files src/

    # Analyze specific concerns
    python pal_analyze.py --prompt "Assess tech debt in the auth module" \\
        --files src/auth/ src/models/user.py

    # Strategic analysis
    python pal_analyze.py --prompt "Should we migrate to microservices?" \\
        --files src/ docs/architecture.md
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response, infer_confidence  # noqa: E402
from conversation import ConversationMemory
from file_utils import read_files
from workflow import AnalyzeWorkflow, AnalyzeWorkflowRequest

from config import load_config
from providers import execute_request, get_provider


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser(description="PAL Analyze - Holistic technical audit and strategic analysis")
    parser.add_argument("--prompt", required=True, help="Analysis question or focus area")
    parser.add_argument("--files", nargs="*", default=[], help="Files/directories to analyze")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing analysis session")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking mode for analysis (default: high)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Workflow parameters for multi-step analysis
    parser.add_argument("--step", help="Analysis step description")
    parser.add_argument("--step-number", type=int, default=1, help="Current step (1-based)")
    parser.add_argument("--total-steps", type=int, default=5, help="Estimated total steps")
    parser.add_argument(
        "--next-step-required",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Continue analysis?",
    )
    parser.add_argument("--findings", help="Discoveries from this step")
    parser.add_argument("--files-checked", nargs="*", default=[], help="All examined files")
    parser.add_argument("--relevant-files", nargs="*", default=[], help="Files relevant to analysis")
    parser.add_argument("--relevant-context", nargs="*", default=[], help="Relevant methods/functions")
    parser.add_argument(
        "--issues-found",
        type=json.loads,
        default=[],
        help='Structured issues as JSON (e.g., \'[{"severity":"high","description":"..."}]\')',
    )
    parser.add_argument(
        "--analysis-type",
        choices=["architecture", "performance", "security", "quality", "general"],
        default="general",
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--output-format",
        choices=["summary", "detailed", "actionable"],
        default="detailed",
        help="Output format preference",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()

        # Initialize conversation memory
        memory = ConversationMemory(config)

        # Check if this is a workflow step (step and findings provided)
        is_workflow_mode = args.step is not None and args.findings is not None

        # Get or create conversation thread
        conversation_history = ""
        if args.continuation_id:
            thread = memory.get_thread(args.continuation_id)
            if thread:
                conversation_history = memory.build_history(thread)
            else:
                thread = memory.create_thread("analyze", {"prompt": args.prompt})
        else:
            thread = memory.create_thread("analyze", {"prompt": args.prompt})

        # Handle workflow mode
        if is_workflow_mode:
            workflow = AnalyzeWorkflow()

            # Create workflow request
            workflow_request = AnalyzeWorkflowRequest(
                prompt=args.prompt,
                step=args.step,
                step_number=args.step_number,
                total_steps=args.total_steps,
                next_step_required=args.next_step_required,
                findings=args.findings,
                files_checked=args.files_checked,
                relevant_files=args.relevant_files,
                relevant_context=args.relevant_context,
                issues_found=args.issues_found,
                analysis_type=args.analysis_type,
                output_format=args.output_format,
                continuation_id=args.continuation_id or "",
                absolute_file_paths=args.files,
            )

            # Accumulate step data
            workflow.accumulate_step_data(workflow_request)

            # Check if we should escalate to codereview
            if workflow.should_call_expert_analysis(workflow.consolidated_findings):
                result = {
                    "status": "full_codereview_required",
                    "important": "Please use pal's codereview tool instead",
                    "reason": "Multiple high-severity issues found requiring detailed code review",
                    "continuation_id": thread["thread_id"],
                    "issues_summary": workflow.consolidated_findings.issues_found,
                }

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n{'=' * 60}")
                    print("Analysis Escalation Required")
                    print(f"{'=' * 60}\n")
                    print("Multiple high-severity issues found.")
                    print("Please use pal's codereview tool for detailed analysis.")
                return

            # Get step guidance if more steps needed
            if args.next_step_required:
                guidance = workflow.get_step_guidance(workflow_request)
                result = workflow.format_step_response(
                    workflow_request,
                    {
                        "guidance": guidance,
                        "continuation_id": thread["thread_id"],
                        "analysis_focus": args.prompt,
                        "analysis_type": args.analysis_type,
                        "output_format": args.output_format,
                    },
                )

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n{'=' * 60}")
                    print(f"Analysis Investigation - Step {args.step_number}/{args.total_steps}")
                    print(f"Type: {args.analysis_type} | Format: {args.output_format}")
                    print(f"{'=' * 60}\n")
                    print(guidance)
                    print(f"\n{'=' * 60}\n")
                return

        # Load system prompt
        system_prompt = load_prompt("analyze")

        # Read files if provided
        file_content = ""
        if args.files:
            file_content = read_files(args.files, include_line_numbers=True)

        # Build user prompt
        user_prompt_parts = []

        if conversation_history:
            user_prompt_parts.append(conversation_history)

        # Add workflow context if in workflow mode
        if is_workflow_mode:
            user_prompt_parts.append("=== ANALYSIS CONTEXT ===")
            user_prompt_parts.append(f"Step {args.step_number} of {args.total_steps}")
            user_prompt_parts.append(f"Analysis type: {args.analysis_type}")
            user_prompt_parts.append(f"Output format: {args.output_format}")
            user_prompt_parts.append(f"Current step: {args.step}")
            user_prompt_parts.append(f"Findings so far: {args.findings}")
            if args.issues_found:
                user_prompt_parts.append(f"Issues found: {len(args.issues_found)}")
            user_prompt_parts.append("")

        user_prompt_parts.append("=== ANALYSIS REQUEST ===")
        user_prompt_parts.append(args.prompt)

        if file_content:
            user_prompt_parts.append("\n=== CODE/PROJECT FILES ===")
            user_prompt_parts.append(file_content)

        user_prompt_parts.append("\n=== INSTRUCTIONS ===")
        user_prompt_parts.append(
            "Provide a holistic technical audit following the analysis framework. "
            "Focus on strategic insights, architectural alignment, and actionable recommendations. "
            "Avoid line-by-line code review - that's for the codereview tool."
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
            temperature=0.7,
            thinking_mode=args.thinking_mode,
            config=config,
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=args.prompt,
            files=args.files if args.files else None,
            tool_name="analyze",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="analyze",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Infer confidence from response
        confidence = infer_confidence(
            response["content"], has_errors=False, has_warnings=True, has_actionable_items=True
        )

        # Build agentic result
        result = build_agentic_response(
            tool_name="analyze",
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
            print(f"\n{'=' * 60}")
            print("Strategic Analysis")
            print(f"Focus: {args.prompt}")
            if args.files:
                print(f"Files: {len(args.files)} analyzed")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Continuation ID: {thread['thread_id']}")
            print(f"{'=' * 60}\n")
            print(response["content"])
            print(f"\n{'=' * 60}")
            if response.get("usage"):
                usage = response["usage"]
                print(f"Tokens: {usage.get('input_tokens', 'N/A')} in / {usage.get('output_tokens', 'N/A')} out")
            # Display agentic metadata
            agentic = result.get("agentic", {})
            print("\n--- Agentic Metadata ---")
            print(f"Confidence: {agentic.get('confidence', 'N/A')}")
            if agentic.get("next_actions"):
                print("Next Actions:")
                for action in agentic["next_actions"]:
                    print(f"  - {action}")
            if agentic.get("related_tools"):
                print(f"Related Tools: {', '.join(agentic['related_tools'])}")
            if agentic.get("escalation_path"):
                print(f"Escalation Path: {agentic['escalation_path']}")
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
