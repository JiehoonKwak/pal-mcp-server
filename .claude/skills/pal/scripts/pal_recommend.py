#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
PAL Recommend - Meta-orchestration tool for task analysis and tool recommendation.

Analyzes task descriptions and recommends appropriate PAL tools with workflow chains.
This tool does NOT require AI calls - it's a simple pattern-based task analyzer.

Run with uv (recommended):
    uv run scripts/pal_recommend.py --task "Task description" [options]

Usage:
    python pal_recommend.py --task "Task description" [options]

Options:
    --task TASK     Task description to analyze (required)
    --json          Output as JSON

Examples:
    # Get recommendations for security task
    python pal_recommend.py --task "Audit code for security vulnerabilities"

    # Get recommendations for debugging
    python pal_recommend.py --task "Fix memory leak in worker process"

    # JSON output
    python pal_recommend.py --task "Review PR quality" --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from agentic import build_agentic_response  # noqa: E402

# Pattern rules: (keywords, tools, workflow, rationale)
# Higher priority rules first
PATTERN_RULES = [
    {
        "patterns": [r"security", r"vulnerab", r"owasp", r"audit", r"cve", r"exploit"],
        "tools": ["secaudit", "codereview"],
        "workflow": "secaudit → fix → testgen → precommit",
        "rationale": "Security-focused workflow for vulnerability assessment",
    },
    {
        "patterns": [r"bug", r"error", r"fix", r"investigat", r"crash", r"fail"],
        "tools": ["debug", "testgen"],
        "workflow": "debug → fix → testgen → precommit",
        "rationale": "Bug investigation and fix workflow with test coverage",
    },
    {
        "patterns": [r"test", r"coverage", r"unit", r"integration", r"e2e"],
        "tools": ["testgen", "codereview"],
        "workflow": "testgen → run tests → codereview",
        "rationale": "Test generation and quality assurance workflow",
    },
    {
        "patterns": [r"document", r"docstring", r"readme", r"jsdoc", r"api\s*doc"],
        "tools": ["docgen"],
        "workflow": "docgen → review → update",
        "rationale": "Documentation generation workflow",
    },
    {
        "patterns": [r"refactor", r"smell", r"clean", r"moderniz", r"legacy"],
        "tools": ["refactor", "codereview"],
        "workflow": "refactor → codereview → testgen → precommit",
        "rationale": "Code modernization with quality checks",
    },
    {
        "patterns": [r"trace", r"flow", r"dependency", r"call\s*(graph|tree|chain)"],
        "tools": ["tracer", "analyze"],
        "workflow": "tracer → analyze → (document or fix)",
        "rationale": "Code flow and dependency analysis",
    },
    {
        "patterns": [r"review", r"quality", r"pr\b", r"pull\s*request"],
        "tools": ["codereview", "precommit"],
        "workflow": "codereview → fix → precommit",
        "rationale": "Code review and pre-commit quality checks",
    },
    {
        "patterns": [r"plan", r"design", r"architect", r"roadmap", r"strateg"],
        "tools": ["planner", "analyze", "thinkdeep"],
        "workflow": "planner → analyze → thinkdeep → (implement)",
        "rationale": "Strategic planning and architectural analysis",
    },
    {
        "patterns": [r"decision", r"perspective", r"opinion", r"which.*better", r"compare"],
        "tools": ["consensus", "thinkdeep"],
        "workflow": "consensus → thinkdeep → (decide)",
        "rationale": "Multi-model consensus for balanced decision making",
    },
    {
        "patterns": [r"complex", r"deep", r"think.*through", r"reason"],
        "tools": ["thinkdeep"],
        "workflow": "thinkdeep → (implement)",
        "rationale": "Deep reasoning for complex problems",
    },
    {
        "patterns": [r"api", r"library", r"docs?\s*lookup", r"how\s*to\s*use"],
        "tools": ["apilookup"],
        "workflow": "apilookup → WebSearch → summarize",
        "rationale": "API and library documentation lookup",
    },
    {
        "patterns": [r"challeng", r"verify", r"validat", r"question", r"critique"],
        "tools": ["challenge"],
        "workflow": "challenge → thinkdeep → (conclude)",
        "rationale": "Critical verification and validation workflow",
    },
    {
        "patterns": [r"models?", r"available", r"list.*model", r"what.*model"],
        "tools": ["listmodels"],
        "workflow": "listmodels → (select model)",
        "rationale": "List available AI models for PAL tools",
    },
]


def match_patterns(task: str, patterns: list[str]) -> int:
    """Count how many patterns match the task (case-insensitive)."""
    task_lower = task.lower()
    return sum(1 for p in patterns if re.search(p, task_lower))


def analyze_task(task: str) -> dict:
    """Analyze task and return recommendations."""
    matches = []

    for rule in PATTERN_RULES:
        match_count = match_patterns(task, rule["patterns"])
        if match_count > 0:
            matches.append((match_count, rule))

    # Sort by match count (highest first)
    matches.sort(key=lambda x: x[0], reverse=True)

    if not matches:
        # Default recommendation for general tasks
        return {
            "recommended": ["chat", "analyze"],
            "workflow": "chat → analyze → (specific tool)",
            "rationale": "General task - start with chat for clarification",
            "confidence": "low",
        }

    best_match = matches[0][1]
    match_count = matches[0][0]

    # Determine confidence based on match strength
    if match_count >= 2:
        confidence = "high"
    elif match_count == 1 and len(matches) == 1:
        confidence = "medium"
    else:
        confidence = "medium"

    return {
        "recommended": best_match["tools"],
        "workflow": best_match["workflow"],
        "rationale": best_match["rationale"],
        "confidence": confidence,
    }


def main():
    parser = argparse.ArgumentParser(description="PAL Recommend - Meta-orchestration tool for task analysis")
    parser.add_argument("--task", required=True, help="Task description to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        # Analyze the task
        analysis = analyze_task(args.task)

        # Build agentic response
        result = build_agentic_response(
            tool_name="recommend",
            status="success",
            content=f"Recommended tools for: {args.task}",
            continuation_id="",
            confidence=analysis["confidence"],
            additional_next_actions=[
                f"Run {analysis['recommended'][0]} on target files",
                "Review findings",
                "Follow the suggested workflow",
            ],
            override_related_tools=analysis["recommended"],
        )

        # Add recommendation details
        result["recommended"] = analysis["recommended"]
        result["workflow"] = analysis["workflow"]
        result["rationale"] = analysis["rationale"]

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'=' * 60}")
            print("PAL Tool Recommendation")
            print(f"{'=' * 60}")
            print(f"Task: {args.task}")
            print(f"\nRecommended Tools: {', '.join(analysis['recommended'])}")
            print(f"Workflow: {analysis['workflow']}")
            print(f"Rationale: {analysis['rationale']}")
            print(f"Confidence: {analysis['confidence']}")
            agentic = result.get("agentic", {})
            if agentic.get("next_actions"):
                print("\nNext Actions:")
                for action in agentic["next_actions"]:
                    print(f"  - {action}")
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
