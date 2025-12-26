# Tool Escalation Patterns

When to escalate from one PAL tool to another based on response metadata.

## Confidence-Based Escalation

When a tool returns `confidence: low` or `confidence: medium`, consider escalating:

| Current Tool | Escalate To | When |
|-------------|-------------|------|
| chat | codereview | Code issues mentioned in response |
| chat | thinkdeep | Complex decisions need deeper analysis |
| codereview | debug | Root cause of issue unclear |
| debug | thinkdeep | Need architectural perspective |
| analyze | codereview | Specific code issues identified |
| consensus | planner | Decision made, need implementation plan |

## Status-Based Escalation

```python
# Check response status
if result["status"] == "needs_escalation":
    # Use the first related_tool
    next_tool = result["agentic"]["related_tools"][0]
```

## Automatic Escalation Triggers

- `status: needs_escalation` - Tool explicitly requests escalation
- `related_tools` non-empty - Tool suggests alternatives
- `confidence: low` after 2+ attempts - Need different approach
- Response mentions "investigate further" or "need more context"

## Escalation Paths

### Code Quality Flow
```
codereview → debug → fix → testgen → precommit → commit
```

### Architecture Decision Flow
```
analyze → thinkdeep → consensus → planner → implement
```

### Bug Investigation Flow
```
debug → (read files) → debug → thinkdeep → fix
```

## Implementation

```python
# After each tool call, check for escalation
result = run_pal_tool("codereview", files=["src/auth.py"])

if result["agentic"]["confidence"] in ["low", "medium"]:
    # Consider escalation
    related = result["agentic"]["related_tools"]
    if "debug" in related:
        # Issue found but root cause unclear
        result = run_pal_tool("debug",
            issue=extract_issue(result["content"]),
            files=result["agentic"]["files_examined"],
            continuation_id=result["continuation_id"])
```
