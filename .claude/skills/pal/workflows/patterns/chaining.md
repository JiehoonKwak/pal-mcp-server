# Tool Chaining Patterns

How to chain multiple PAL tools for complex workflows.

## Basic Chain Pattern

```bash
# 1. Start with analysis
result=$(uv run scripts/pal_analyze.py --prompt "Assess architecture" --files src/ --json)
CID=$(echo $result | jq -r '.continuation_id')

# 2. Deep dive on findings
result=$(uv run scripts/pal_thinkdeep.py --prompt "Analyze identified issues" \
  --continuation-id $CID --json)

# 3. Get multiple perspectives
result=$(uv run scripts/pal_consensus.py --proposal "Recommended changes from analysis" \
  --models gemini-2.5-flash gpt-4o --json)
```

## Common Chains

### Code Review Chain
```
codereview → (fix issues) → testgen → precommit → commit
```
1. Review code for issues
2. Fix identified problems
3. Generate tests for changes
4. Validate with precommit
5. Commit if passing

### Debug Chain
```
debug → (read files) → debug → (implement fix) → testgen
```
1. Initial debugging analysis
2. Read identified suspicious files
3. Continue debug with more context
4. Implement the fix
5. Generate regression tests

### Architecture Decision Chain
```
analyze → thinkdeep → consensus → planner
```
1. Analyze current state
2. Deep analysis of options
3. Get multi-model perspectives
4. Create implementation plan

## Parallel Chains

For independent analyses, run in parallel:

```bash
# Run security and performance reviews in parallel
uv run scripts/pal_secaudit.py --files src/ --json &
uv run scripts/pal_codereview.py --files src/ --focus performance --json &
wait
```

## Chain State Management

All tools share state via `continuation_id`:

```python
# Start chain
result1 = run_tool("analyze", files=["src/"])
cid = result1["continuation_id"]

# Continue with same context
result2 = run_tool("codereview", files=["src/"], continuation_id=cid)

# Files from both calls available
result3 = run_tool("debug", issue="...", continuation_id=cid)
```

## Conditional Chains

Use response metadata to decide next step:

```python
result = run_tool("codereview", files=files)

if "security" in result["content"].lower():
    # Security issue found, escalate
    run_tool("secaudit", files=files, continuation_id=result["continuation_id"])
elif result["agentic"]["confidence"] == "low":
    # Need deeper analysis
    run_tool("thinkdeep", prompt="Analyze review findings",
             continuation_id=result["continuation_id"])
else:
    # Proceed to implementation
    apply_fixes(result)
```
