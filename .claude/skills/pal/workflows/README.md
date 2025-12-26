# PAL Skill Workflows

Guide for Claude to orchestrate multi-tool workflows using PAL skill responses.

## Agentic Architecture

PAL skills return rich metadata enabling Claude-driven orchestration:

```json
{
  "status": "success",
  "content": "...",
  "continuation_id": "uuid",
  "agentic": {
    "confidence": "high",
    "next_actions": ["Fix identified issues", "Run tests"],
    "related_tools": ["debug", "testgen", "precommit"],
    "escalation_path": "codereview → debug → fix → testgen"
  }
}
```

## Confidence-Based Decisions

| Confidence | Action |
|------------|--------|
| `low` | Gather more context, use a `related_tool`, or escalate |
| `medium` | Proceed with caution, consider second opinion |
| `high` | Implement recommendations |
| `certain` | Direct implementation, high trust |

## Tool Selection Guide

| Task Type | Start With | Escalate To |
|-----------|------------|-------------|
| Quick question | chat | thinkdeep |
| Code issues | codereview | debug → testgen |
| Bug investigation | debug | thinkdeep |
| Architecture decision | thinkdeep | consensus |
| Multiple perspectives | consensus | planner |
| Implementation planning | planner | (implement) |

## Cross-Tool Continuation

All tools share `continuation_id` for context preservation:

```bash
# Start with codereview
uv run scripts/pal_codereview.py --files src/auth.py --json
# Returns: {"continuation_id": "abc-123"}

# Escalate to debug with same context
uv run scripts/pal_debug.py --issue "Investigate SQL issue from review" \
  --files src/auth.py --continuation-id abc-123
```

## Pattern Reference

- [Escalation Patterns](patterns/escalation.md) - When to switch tools
- [Tool Chaining](patterns/chaining.md) - Multi-tool workflows
- [Continuation](patterns/continuation.md) - Multi-turn conversations

## Example Workflows

- [Bug Investigation](examples/bug-investigation.md)
- [Security Audit](examples/security-audit.md)
- [Code Review Workflow](examples/code-review-workflow.md)
