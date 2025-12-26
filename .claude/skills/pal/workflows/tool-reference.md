# PAL Tool Reference

Quick reference for all PAL skill tools with agentic metadata.

## Tool Overview

| Tool | Purpose | Default Confidence | Escalation Path |
|------|---------|-------------------|-----------------|
| **chat** | Brainstorming, questions | medium | → codereview → thinkdeep |
| **thinkdeep** | Deep analysis | high | → planner → implement |
| **codereview** | Code quality review | high | → debug → testgen → precommit |
| **debug** | Bug investigation | medium | → testgen → codereview |
| **analyze** | Architecture assessment | medium | → codereview → refactor |
| **consensus** | Multi-model perspectives | high | → thinkdeep → planner |
| **planner** | Implementation planning | high | → execute steps |
| **clink** | External CLI bridge | medium | varies |
| **secaudit** | Security audit | high | → fix → testgen → precommit |
| **testgen** | Test generation | high | → run tests → precommit |
| **docgen** | Documentation | high | → review |
| **refactor** | Code refactoring | medium | → codereview → testgen |
| **precommit** | Pre-commit validation | high | → fix → commit |
| **tracer** | Call path analysis | medium | → debug → fix |
| **challenge** | Critical analysis | high | - |
| **apilookup** | API documentation | medium | - |
| **listmodels** | Available models | certain | - |

## Tool Details

### Chat
```bash
uv run scripts/pal_chat.py --prompt "Your question" \
  [--files FILE ...] [--model MODEL] [--continuation-id ID]
```
**Use for**: Quick questions, brainstorming, general discussion
**Related tools**: codereview, thinkdeep, consensus

### ThinkDeep
```bash
uv run scripts/pal_thinkdeep.py --prompt "Complex question" \
  [--files FILE ...] [--thinking-mode high|max]
```
**Use for**: Architecture decisions, trade-off analysis, complex problems
**Default thinking-mode**: high

### CodeReview
```bash
uv run scripts/pal_codereview.py --files FILE [FILE ...] \
  [--focus security|performance|maintainability|architecture|testing]
```
**Use for**: Code quality, bug detection, security review
**Focus areas**: security, performance, maintainability, architecture, testing

### Debug
```bash
uv run scripts/pal_debug.py --issue "Bug description" --files FILE [FILE ...] \
  [--error-logs LOGS] [--thinking-mode high]
```
**Use for**: Bug investigation, root cause analysis, hypothesis testing
**Features**: Workflow support with step tracking and confidence levels

### Analyze
```bash
uv run scripts/pal_analyze.py --prompt "Analysis question" --files FILE [FILE ...] \
  [--analysis-type architecture|performance|security|quality|general]
```
**Use for**: Architecture assessment, tech debt evaluation, scalability analysis

### Consensus
```bash
uv run scripts/pal_consensus.py --proposal "Decision question" \
  --models MODEL1 MODEL2 [MODEL3] \
  [--stances for against neutral] [--files FILE ...]
```
**Use for**: Multiple perspectives, high-stakes decisions, controversial proposals
**Requires**: At least 2 models

### Planner
```bash
uv run scripts/pal_planner.py --task "Feature to implement" \
  [--files FILE ...] [--thinking-mode high]
```
**Use for**: Implementation planning, phase breakdown, risk identification

### Clink
```bash
uv run scripts/pal_clink.py --cli gemini|claude|codex --prompt "Request" \
  [--role default|planner|codereviewer] [--files FILE ...]
```
**Use for**: Leveraging external AI CLIs, specific CLI capabilities

## Response Structure

All tools return:
```json
{
  "status": "success|error|needs_escalation",
  "content": "Response text",
  "continuation_id": "uuid-for-continuation",
  "model": "model-used",
  "provider": "provider-name",
  "agentic": {
    "confidence": "low|medium|high|certain",
    "next_actions": ["Suggested action 1", "Action 2"],
    "related_tools": ["tool1", "tool2"],
    "escalation_path": "tool1 → tool2 → tool3"
  },
  "usage": {"input_tokens": N, "output_tokens": N}
}
```
