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
PAL Planner - Implementation planning with multi-step workflow support.

Provides structured planning for complex tasks with phase-based breakdown,
dependency tracking, risk identification, and validation gates.

Run with uv (recommended):
    uv run scripts/pal_planner.py --task "Task description" --files FILE [options]

Usage:
    python pal_planner.py --task "Task description" --files FILE [options]

Options:
    --task TASK              Task or feature to plan (required)
    --files FILE [...]       Files/directories to consider
    --model MODEL            Model to use (default: from config)
    --continuation-id ID     Continue existing planning session

Examples:
    # Plan a new feature
    python pal_planner.py --task "Add user authentication with OAuth2" \\
        --files src/

    # Plan with specific context
    python pal_planner.py --task "Migrate database from SQLite to PostgreSQL" \\
        --files src/models/ config/

    # Continue planning session
    python pal_planner.py --task "Continue with phase 2" \\
        --continuation-id <uuid>
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from conversation import ConversationMemory
from file_utils import read_files
from workflow import ConsolidatedFindings, WorkflowRequest, WorkflowTool

from config import load_config
from providers import execute_request, get_provider


@dataclass
class PlannerWorkflowRequest(WorkflowRequest):
    """Planner-specific workflow request."""

    current_phase: str = ""
    dependencies: list[str] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    validation_gates: list[str] = field(default_factory=list)
    plan_summary: str = ""


class PlannerWorkflow(WorkflowTool):
    """Planner-specific workflow implementation."""

    def get_name(self) -> str:
        return "planner"

    def get_system_prompt(self) -> str:
        """Load the planner system prompt."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "planner.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        # Fallback system prompt if file doesn't exist
        return """You are an expert software architect and planner.

Your role is to create detailed, actionable implementation plans for complex tasks.

For each plan:
1. Break down the work into numbered phases with clear deliverables
2. Identify dependencies between phases
3. Highlight risks and propose mitigations
4. Define validation gates for each phase
5. Provide specific, actionable next steps

Keep plans concise and focused on what another engineer needs to execute.
Avoid lengthy explanations - focus on the essentials."""

    def get_required_actions(self, step_number: int, confidence: str, findings: str, total_steps: int) -> list[str]:
        """Get required actions for planning steps."""
        if step_number == 1:
            return [
                "Analyze the task requirements and constraints",
                "Identify key files and components that will be affected",
                "Understand existing patterns and architecture",
                "Note any potential blockers or dependencies",
            ]
        elif step_number <= total_steps // 2:
            return [
                "Define phases and their deliverables",
                "Map dependencies between phases",
                "Identify risks for each phase",
                "Set validation criteria",
            ]
        else:
            return [
                "Finalize the plan structure",
                "Ensure all phases have clear next actions",
                "Verify risk mitigations are adequate",
                "Prepare the executive summary",
            ]

    def should_call_expert_analysis(self, consolidated_findings: ConsolidatedFindings) -> bool:
        """Planner doesn't need expert escalation."""
        return False

    def accumulate_step_data(self, request: PlannerWorkflowRequest) -> None:
        """Accumulate planner-specific findings."""
        super().accumulate_step_data(request)

        # Track risks and dependencies
        if hasattr(request, "risks") and request.risks:
            self.consolidated_findings.issues_found.extend({"type": "risk", **r} for r in request.risks)


def load_prompt(prompt_name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def main():
    parser = argparse.ArgumentParser(description="PAL Planner - Implementation planning with workflow support")
    parser.add_argument("--task", required=True, help="Task or feature to plan")
    parser.add_argument("--files", nargs="*", default=[], help="Files/directories to consider")
    parser.add_argument("--model", help="Model to use (default: from config)")
    parser.add_argument("--continuation-id", help="Continue existing planning session")
    parser.add_argument(
        "--thinking-mode",
        choices=["minimal", "low", "medium", "high", "max"],
        default="high",
        help="Thinking mode for planning (default: high)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Workflow parameters for multi-step planning
    parser.add_argument("--step", help="Planning step description")
    parser.add_argument("--step-number", type=int, default=1, help="Current step (1-based)")
    parser.add_argument("--total-steps", type=int, default=5, help="Estimated total steps")
    parser.add_argument(
        "--next-step-required",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Continue planning?",
    )
    parser.add_argument("--findings", help="Discoveries from this step")
    parser.add_argument("--files-checked", nargs="*", default=[], help="All examined files")
    parser.add_argument("--relevant-files", nargs="*", default=[], help="Files relevant to plan")
    parser.add_argument("--relevant-context", nargs="*", default=[], help="Relevant components")
    parser.add_argument("--current-phase", help="Current planning phase name")
    parser.add_argument(
        "--dependencies",
        type=json.loads,
        default=[],
        help='Phase dependencies as JSON (e.g., \'["phase1", "phase2"]\')',
    )
    parser.add_argument(
        "--risks",
        type=json.loads,
        default=[],
        help='Risks as JSON (e.g., \'[{"name":"risk1","mitigation":"..."}]\')',
    )
    parser.add_argument(
        "--validation-gates",
        type=json.loads,
        default=[],
        help='Validation gates as JSON (e.g., \'["gate1", "gate2"]\')',
    )
    parser.add_argument("--plan-summary", help="Final plan summary")

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
                thread = memory.create_thread("planner", {"task": args.task})
        else:
            thread = memory.create_thread("planner", {"task": args.task})

        # Handle workflow mode
        if is_workflow_mode:
            workflow = PlannerWorkflow()

            # Create workflow request
            workflow_request = PlannerWorkflowRequest(
                prompt=args.task,
                step=args.step,
                step_number=args.step_number,
                total_steps=args.total_steps,
                next_step_required=args.next_step_required,
                findings=args.findings,
                files_checked=args.files_checked,
                relevant_files=args.relevant_files,
                relevant_context=args.relevant_context,
                current_phase=args.current_phase or "",
                dependencies=args.dependencies,
                risks=args.risks,
                validation_gates=args.validation_gates,
                plan_summary=args.plan_summary or "",
                continuation_id=args.continuation_id or "",
                absolute_file_paths=args.files,
            )

            # Accumulate step data
            workflow.accumulate_step_data(workflow_request)

            # Check if planning is complete
            if not args.next_step_required and args.plan_summary:
                result = {
                    "status": "planning_complete",
                    "continuation_id": thread["thread_id"],
                    "task": args.task,
                    "plan_summary": args.plan_summary,
                    "phases": {
                        "total_steps": args.step_number,
                        "current_phase": args.current_phase,
                        "dependencies": args.dependencies,
                        "risks": args.risks,
                        "validation_gates": args.validation_gates,
                    },
                }

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n{'='*60}")
                    print("Planning Complete")
                    print(f"Task: {args.task}")
                    print(f"{'='*60}\n")
                    print(args.plan_summary)
                    if args.risks:
                        print("\n--- Identified Risks ---")
                        for risk in args.risks:
                            print(f"- {risk.get('name', 'Unknown')}: {risk.get('mitigation', 'No mitigation')}")
                    print(f"\n{'='*60}\n")
                return

            # Get step guidance if more steps needed
            if args.next_step_required:
                guidance = workflow.get_step_guidance(workflow_request)
                result = workflow.format_step_response(
                    workflow_request,
                    {
                        "guidance": guidance,
                        "continuation_id": thread["thread_id"],
                        "task": args.task,
                        "current_phase": args.current_phase,
                    },
                )

                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print(f"\n{'='*60}")
                    print(f"Planning - Step {args.step_number}/{args.total_steps}")
                    if args.current_phase:
                        print(f"Phase: {args.current_phase}")
                    print(f"{'='*60}\n")
                    print(guidance)
                    print(f"\n{'='*60}\n")
                return

        # Load system prompt
        system_prompt = load_prompt("planner")
        if not system_prompt:
            # Use the clink planner prompt as fallback
            system_prompt = load_prompt("clink/planner")

        # Use default if still empty
        if not system_prompt:
            workflow = PlannerWorkflow()
            system_prompt = workflow.get_system_prompt()

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
            user_prompt_parts.append("=== PLANNING CONTEXT ===")
            user_prompt_parts.append(f"Step {args.step_number} of {args.total_steps}")
            if args.current_phase:
                user_prompt_parts.append(f"Current phase: {args.current_phase}")
            user_prompt_parts.append(f"Current step: {args.step}")
            user_prompt_parts.append(f"Findings so far: {args.findings}")
            if args.dependencies:
                user_prompt_parts.append(f"Dependencies: {', '.join(args.dependencies)}")
            if args.risks:
                user_prompt_parts.append(f"Risks identified: {len(args.risks)}")
            user_prompt_parts.append("")

        user_prompt_parts.append("=== PLANNING REQUEST ===")
        user_prompt_parts.append(args.task)

        if file_content:
            user_prompt_parts.append("\n=== RELEVANT FILES ===")
            user_prompt_parts.append(file_content)

        user_prompt_parts.append("\n=== INSTRUCTIONS ===")
        user_prompt_parts.append(
            "Create a detailed implementation plan with:\n"
            "1. Numbered phases with clear deliverables\n"
            "2. Dependencies between phases\n"
            "3. Risks and mitigations\n"
            "4. Validation gates for each phase\n"
            "5. Specific, actionable next steps"
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
        )

        # Record turns
        memory.add_turn(
            thread_id=thread["thread_id"],
            role="user",
            content=f"Plan: {args.task}",
            files=args.files if args.files else None,
            tool_name="planner",
        )

        memory.add_turn(
            thread_id=thread["thread_id"],
            role="assistant",
            content=response["content"],
            tool_name="planner",
            model_name=resolved_model,
            model_provider=provider.provider_name,
        )

        # Build result
        result = {
            "status": "success",
            "content": response["content"],
            "continuation_id": thread["thread_id"],
            "task": args.task,
            "files_analyzed": args.files if args.files else None,
            "model": resolved_model,
            "provider": provider.provider_name,
            "usage": response.get("usage", {}),
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print("Implementation Plan")
            print(f"Task: {args.task}")
            if args.files:
                print(f"Files: {len(args.files)} analyzed")
            print(f"Model: {resolved_model} via {provider.provider_name}")
            print(f"Continuation ID: {thread['thread_id']}")
            print(f"{'='*60}\n")
            print(response["content"])
            print(f"\n{'='*60}")
            if response.get("usage"):
                usage = response["usage"]
                print(f"Tokens: {usage.get('input_tokens', 'N/A')} in / " f"{usage.get('output_tokens', 'N/A')} out")
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
