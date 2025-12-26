# Agentic Orchestration Workflow

How Claude should orchestrate PAL tools based on response metadata.

## Response Schema

Every PAL tool returns:
```json
{
  "status": "success|error",
  "content": "Response text",
  "continuation_id": "uuid",
  "agentic": {
    "confidence": "low|medium|high|certain",
    "next_actions": ["Action 1", "Action 2"],
    "related_tools": ["tool1", "tool2"],
    "escalation_path": "tool1 → tool2 → tool3"
  }
}
```

## Confidence-Based Routing

### 1. Parse Response
```python
result = json.loads(output)
confidence = result["agentic"]["confidence"]
```

### 2. Route by Confidence

| Confidence | Action |
|------------|--------|
| **certain** | Proceed with implementation |
| **high** | Proceed, verify critical details |
| **medium** | Consider `related_tools` or gather more context |
| **low** | Follow `escalation_path` or add files |

### 3. Escalation Decision
```
IF confidence == "low":
    → Use first tool in escalation_path
    → Pass continuation_id to preserve context

IF confidence == "medium" AND task is critical:
    → Use Consensus for multiple perspectives
    → OR use ThinkDeep for deeper analysis
```

## Escalation Patterns

### Debug → Fix Flow
```
debug (hypothesis) → [low confidence] → add more files → debug again
debug (hypothesis) → [high confidence] → implement fix → testgen → precommit
```

### Review → Ship Flow
```
codereview → [issues found] → debug → fix → testgen → precommit → commit
codereview → [no issues] → precommit → commit
```

### Architecture Decision Flow
```
thinkdeep → [medium confidence] → consensus (multi-model) → planner → implement
```

## Continuation Chaining

Cross-tool continuation preserves context:
```bash
# Step 1: Initial review
uv run scripts/pal_codereview.py --files src/auth.py --json
# Returns: {"continuation_id": "abc-123"}

# Step 2: Deep dive on finding (uses same context)
uv run scripts/pal_thinkdeep.py \
  --prompt "Analyze the SQL injection risk" \
  --continuation-id abc-123 --json

# Step 3: Debug the issue (still has full context)
uv run scripts/pal_debug.py \
  --issue "Fix the SQL injection" \
  --continuation-id abc-123 --json
```

## Parallel Consultation

For high-stakes decisions, run tools in parallel:
```bash
# Parallel: Get multiple model perspectives
uv run scripts/pal_consensus.py \
  --proposal "Should we refactor auth?" \
  --models gemini-2.5-pro gpt-4o grok-3
```

## When to Escalate

| Situation | Escalate To |
|-----------|-------------|
| Bug not reproducible | Debug with more files/logs |
| Security concern | SecAudit → ThinkDeep |
| Architecture question | ThinkDeep → Consensus |
| Fix needs validation | TestGen → Precommit |
| Contentious decision | Consensus with stances |
