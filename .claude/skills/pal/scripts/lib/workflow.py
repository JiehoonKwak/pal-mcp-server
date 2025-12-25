"""
Workflow base class for multi-step investigation tools.

Provides infrastructure for tools that require iterative investigation
with structured findings accumulation and expert analysis escalation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class WorkflowRequest:
    """Base request for workflow tools."""

    prompt: str
    step: str
    step_number: int
    total_steps: int
    next_step_required: bool
    findings: str
    files_checked: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    relevant_context: list[str] = field(default_factory=list)
    model: str = "auto"
    continuation_id: str = ""
    absolute_file_paths: list[str] = field(default_factory=list)


@dataclass
class DebugWorkflowRequest(WorkflowRequest):
    """Debug-specific workflow request with hypothesis tracking."""

    hypothesis: str = ""
    confidence: Literal["exploring", "low", "medium", "high", "very_high", "almost_certain", "certain"] = "exploring"


@dataclass
class AnalyzeWorkflowRequest(WorkflowRequest):
    """Analyze-specific workflow request."""

    issues_found: list[dict] = field(default_factory=list)
    analysis_type: Literal["architecture", "performance", "security", "quality", "general"] = "general"
    output_format: Literal["summary", "detailed", "actionable"] = "detailed"


@dataclass
class ConsolidatedFindings:
    """Accumulated findings across workflow steps."""

    findings: list[str] = field(default_factory=list)
    files_checked: set = field(default_factory=set)
    relevant_files: set = field(default_factory=set)
    relevant_context: list[str] = field(default_factory=list)
    issues_found: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    confidence: str = "low"


class WorkflowTool(ABC):
    """
    Base class for multi-step workflow tools.

    Workflow tools support iterative investigation with:
    - Step tracking (step_number, total_steps, next_step_required)
    - Findings accumulation across steps
    - Confidence level tracking
    - Expert analysis escalation
    """

    def __init__(self):
        self.consolidated_findings = ConsolidatedFindings()

    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for the tool."""
        pass

    @abstractmethod
    def get_required_actions(self, step_number: int, confidence: str, findings: str, total_steps: int) -> list[str]:
        """Get required actions based on current step and confidence."""
        pass

    @abstractmethod
    def should_call_expert_analysis(self, consolidated_findings: ConsolidatedFindings) -> bool:
        """Determine if expert analysis should be invoked."""
        pass

    def accumulate_step_data(self, request: WorkflowRequest) -> None:
        """Accumulate findings from current step."""
        self.consolidated_findings.findings.append(f"Step {request.step_number}: {request.findings}")
        self.consolidated_findings.files_checked.update(request.files_checked)
        self.consolidated_findings.relevant_files.update(request.relevant_files)
        self.consolidated_findings.relevant_context.extend(request.relevant_context)

    def get_step_guidance(self, request: WorkflowRequest) -> str:
        """Generate guidance for next investigation step."""
        confidence = getattr(request, "confidence", "low")
        actions = self.get_required_actions(request.step_number, confidence, request.findings, request.total_steps)

        if request.step_number == 1:
            return (
                f"MANDATORY: DO NOT call {self.get_name()} again immediately. "
                f"You MUST first investigate using appropriate tools. "
                f"Required actions:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
            )
        elif request.next_step_required:
            return f"PAUSE! Before calling {self.get_name()} step {request.step_number + 1}:\n" + "\n".join(
                f"{i+1}. {a}" for i, a in enumerate(actions)
            )
        else:
            return "Investigation complete. Proceeding to expert analysis."

    def prepare_expert_context(self) -> str:
        """Prepare context for expert analysis."""
        parts = [
            "=== INVESTIGATION SUMMARY ===",
            "",
            "**Steps Completed:**",
            "\n".join(self.consolidated_findings.findings),
            "",
            "**Files Examined:**",
            "\n".join(f"- {f}" for f in sorted(self.consolidated_findings.files_checked)),
            "",
            "**Relevant Files:**",
            "\n".join(f"- {f}" for f in sorted(self.consolidated_findings.relevant_files)),
            "",
        ]

        if self.consolidated_findings.relevant_context:
            parts.extend(
                [
                    "**Key Context:**",
                    "\n".join(f"- {c}" for c in self.consolidated_findings.relevant_context),
                    "",
                ]
            )

        if self.consolidated_findings.issues_found:
            parts.extend(
                [
                    "**Issues Found:**",
                    *[
                        f"- [{i.get('severity', 'unknown')}] {i.get('description', 'No description')}"
                        for i in self.consolidated_findings.issues_found
                    ],
                    "",
                ]
            )

        if self.consolidated_findings.hypotheses:
            parts.extend(
                [
                    "**Hypotheses:**",
                    *[
                        f"- [{h.get('confidence', 'low')}] {h.get('name', 'Unknown')}: {h.get('description', '')}"
                        for h in self.consolidated_findings.hypotheses
                    ],
                    "",
                ]
            )

        return "\n".join(parts)

    def format_step_response(self, request: WorkflowRequest, additional_data: Optional[dict] = None) -> dict:
        """Format response for a workflow step."""
        response = {
            "status": "investigation_in_progress" if request.next_step_required else "investigation_complete",
            "step": request.step_number,
            "total_steps": request.total_steps,
            "next_step_required": request.next_step_required,
            "investigation_status": {
                "steps_completed": request.step_number,
                "files_examined": len(self.consolidated_findings.files_checked),
                "relevant_files_found": len(self.consolidated_findings.relevant_files),
            },
        }

        if request.next_step_required:
            response["guidance"] = self.get_step_guidance(request)

        if additional_data:
            response.update(additional_data)

        return response


class DebugWorkflow(WorkflowTool):
    """Debug-specific workflow implementation."""

    def get_name(self) -> str:
        return "debug"

    def get_system_prompt(self) -> str:
        from pathlib import Path

        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "debug.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    def get_required_actions(self, step_number: int, confidence: str, findings: str, total_steps: int) -> list[str]:
        """Get required actions based on confidence level."""
        if confidence in ["exploring", "low"]:
            return [
                "Read and analyze the mentioned files",
                "Look for patterns related to the reported symptoms",
                "Document specific code locations of interest",
                "Form initial hypotheses based on evidence",
            ]
        elif confidence == "medium":
            return [
                "Verify the leading hypothesis against actual code",
                "Check for edge cases and error handling",
                "Review related test files if they exist",
                "Consider alternative explanations",
            ]
        elif confidence in ["high", "very_high"]:
            return [
                "Validate the root cause with specific code evidence",
                "Determine the minimal fix required",
                "Assess potential regression risks",
                "Document the complete investigation trail",
            ]
        elif confidence in ["almost_certain", "certain"]:
            return [
                "Finalize the fix recommendation",
                "Complete the investigation summary",
            ]
        return []

    def should_call_expert_analysis(self, consolidated_findings: ConsolidatedFindings) -> bool:
        """Debug should call expert when confidence is high but not certain."""
        return consolidated_findings.confidence in ["high", "very_high", "almost_certain"]

    def accumulate_step_data(self, request: DebugWorkflowRequest) -> None:
        """Accumulate debug-specific findings."""
        super().accumulate_step_data(request)
        self.consolidated_findings.confidence = request.confidence

        if request.hypothesis:
            self.consolidated_findings.hypotheses.append(
                {
                    "name": f"Hypothesis Step {request.step_number}",
                    "description": request.hypothesis,
                    "confidence": request.confidence,
                }
            )


class AnalyzeWorkflow(WorkflowTool):
    """Analyze-specific workflow implementation."""

    def get_name(self) -> str:
        return "analyze"

    def get_system_prompt(self) -> str:
        from pathlib import Path

        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "analyze.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    def get_required_actions(self, step_number: int, confidence: str, findings: str, total_steps: int) -> list[str]:
        """Get required actions for analysis steps."""
        if step_number <= total_steps // 2:
            return [
                "Examine the codebase structure and architecture",
                "Identify key modules and their responsibilities",
                "Document dependencies and coupling patterns",
                "Note areas of concern or technical debt",
            ]
        else:
            return [
                "Synthesize findings into actionable insights",
                "Prioritize issues by impact and effort",
                "Formulate specific recommendations",
                "Prepare the executive summary",
            ]

    def should_call_expert_analysis(self, consolidated_findings: ConsolidatedFindings) -> bool:
        """Analyze should escalate to codereview for detailed issues."""
        # Escalate if many high-severity issues found
        high_severity = sum(1 for i in consolidated_findings.issues_found if i.get("severity") in ["high", "critical"])
        return high_severity >= 3

    def accumulate_step_data(self, request: AnalyzeWorkflowRequest) -> None:
        """Accumulate analyze-specific findings."""
        super().accumulate_step_data(request)
        self.consolidated_findings.issues_found.extend(request.issues_found)
