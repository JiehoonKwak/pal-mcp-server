# Bug Investigation Workflow

Systematic debugging using PAL tools with escalation.

## Workflow Steps

```
debug → (read files) → debug → (verify hypothesis) → fix → testgen → precommit
```

## Example

### 1. Initial Debug Analysis
```bash
result=$(uv run scripts/pal_debug.py \
  --issue "API returns 500 on login with valid credentials" \
  --files src/auth.py src/api/routes.py \
  --error-logs "$(cat logs/error.log)" \
  --json)

# Check response
echo $result | jq '.agentic.confidence'
# Output: "medium"

CID=$(echo $result | jq -r '.continuation_id')
```

### 2. Read Identified Files
Based on debug response, read suspicious files:
```bash
# Debug identified src/auth.py:45-60 and src/db/connection.py as suspicious
cat src/db/connection.py
```

### 3. Continue Debug with More Context
```bash
result=$(uv run scripts/pal_debug.py \
  --issue "Verify database connection timeout hypothesis" \
  --files src/auth.py src/db/connection.py \
  --continuation-id $CID \
  --json)

echo $result | jq '.agentic.confidence'
# Output: "high"
```

### 4. Escalate to ThinkDeep (if needed)
If confidence still medium:
```bash
uv run scripts/pal_thinkdeep.py \
  --prompt "Deep analysis of connection pooling issues" \
  --files src/db/connection.py \
  --continuation-id $CID \
  --thinking-mode max
```

### 5. Implement Fix
Based on confirmed root cause, implement the fix.

### 6. Generate Regression Tests
```bash
uv run scripts/pal_testgen.py \
  --files src/db/connection.py \
  --prompt "Generate tests for connection timeout handling" \
  --continuation-id $CID
```

### 7. Pre-commit Validation
```bash
uv run scripts/pal_precommit.py \
  --files src/db/connection.py tests/test_connection.py \
  --continuation-id $CID
```

## Agentic Decision Points

| After | Check | Action |
|-------|-------|--------|
| debug (1st) | confidence == "low" | Read more files, gather context |
| debug (2nd) | confidence == "medium" | Continue or escalate to thinkdeep |
| debug | confidence == "high" | Proceed to implement fix |
| fix | - | Generate tests |
| testgen | tests pass | Run precommit |
| precommit | all pass | Commit changes |
