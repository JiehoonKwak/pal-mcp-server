---
name: pal
description: "PAL (Provider Abstraction Layer) - Multi-model AI development assistant with conversation memory. Use for code review, debugging, deep analysis, planning, and collaborative thinking. Supports Gemini, OpenAI, XAI, and local models. Features multi-turn conversations with continuation_id and CLI bridging to spawn Gemini/Claude/Codex CLIs. Activate for second opinions, expert code review, architectural decisions, or multi-model consensus."
allowed-tools: [Read, Glob, Grep, Edit, Write, Bash, Task]
---

# PAL - Provider Abstraction Layer Skills

PAL provides access to multiple AI models (Gemini, OpenAI, XAI, OpenRouter, Ollama) for specialized development tasks with conversation memory and cross-tool continuation.

## Quick Start

```bash
# Set your API key (at least one required)
export GEMINI_API_KEY="your-key"

# Or configure in config/config.yaml

# Run with uv (recommended - no install needed)
uv run scripts/pal_chat.py --prompt "Hello, PAL!" --json

# Or install dependencies first
pip install -r requirements.txt
python scripts/pal_chat.py --prompt "Hello, PAL!" --json
```

## When to Use Each Tool

Choose the right tool for your task:

| Task | Tool | Why |
|------|------|-----|
| Simple questions, brainstorming | **Chat** | Fast responses, general discussion |
| Complex architecture decisions | **ThinkDeep** | Extended reasoning with high thinking budget |
| Code quality before PR/commit | **CodeReview** | Structured review with severity ratings |
| Bug investigation | **Debug** | Systematic debugging with hypothesis generation |
| Strategic technical audit | **Analyze** | High-level architecture and scalability analysis |
| Multiple perspectives needed | **Consensus** | Consult 2+ models with optional stance steering |
| Leverage other CLI tools | **Clink** | Bridge to Gemini/Claude/Codex CLIs |

### Detailed Use Cases

**Use Chat when:**
- You need a quick second opinion
- Brainstorming ideas or approaches
- General development questions
- Validating a simple decision

**Use ThinkDeep when:**
- Facing complex architectural decisions
- Need deep analysis of trade-offs
- Investigating performance implications
- Security threat modeling
- Design pattern selection

**Use CodeReview when:**
- Before creating a PR
- After major refactoring
- Security-focused code audit
- Performance optimization review
- Onboarding to unfamiliar code

**Use Debug when:**
- Investigating production bugs
- Analyzing error logs and stack traces
- Hypothesis-driven debugging
- Understanding complex failure modes

**Use Analyze when:**
- Evaluating technical debt
- Assessing architecture scalability
- Strategic technology decisions
- Long-term maintainability review

**Use Consensus when:**
- Need diverse perspectives
- Evaluating contentious proposals
- Making high-stakes decisions
- Want pro/con analysis from different models

**Use Clink when:**
- Need Gemini's web search capability
- Want Claude's extended context
- Leveraging specific CLI features

## Available Tools

All tools support PEP 723 inline metadata - run directly with `uv run`:

### Chat - Collaborative Thinking

General development chat with multi-turn conversation support.

```bash
uv run scripts/pal_chat.py \
  --prompt "Your question or discussion topic" \
  --files path/to/file.py \
  --model gemini-2.5-flash \
  --continuation-id <uuid-for-multi-turn>
```

**Options:**
- `--prompt` (required): Your question or topic
- `--files`: Files to include as context
- `--images`: Images to include (paths or data URLs)
- `--model`: Model to use (default: auto)
- `--continuation-id`: Continue existing conversation
- `--thinking-mode`: minimal/low/medium/high/max
- `--json`: Output as JSON

### ThinkDeep - Deep Analysis

Extended thinking for complex problems with higher reasoning budget.

```bash
uv run scripts/pal_thinkdeep.py \
  --prompt "Complex architectural question" \
  --files src/core.py \
  --thinking-mode high
```

**Defaults to `--thinking-mode high`** for deeper analysis.

### CodeReview - Expert Code Review

Comprehensive multi-pass code review with severity ratings.

```bash
uv run scripts/pal_codereview.py \
  --files src/main.py src/utils.py \
  --focus security performance \
  --model gemini-2.5-pro
```

**Focus areas:** security, performance, maintainability, architecture, testing

### Debug - Expert Debugging

Systematic debugging with hypothesis generation and root cause analysis.

```bash
uv run scripts/pal_debug.py \
  --issue "API returns 500 on login" \
  --files src/auth.py src/api.py \
  --error-logs "Stack trace here..."
```

**Options:**
- `--issue` (required): Description of the bug/issue
- `--files` (required): Files to analyze
- `--error-logs`: Error logs or stack traces
- `--thinking-mode`: defaults to high for thorough analysis

### Analyze - Strategic Analysis

Holistic technical audit focusing on architecture, scalability, and strategic improvements.

```bash
uv run scripts/pal_analyze.py \
  --prompt "Is this architecture scalable?" \
  --files src/ docs/architecture.md
```

**Use for:** Architecture assessment, tech debt evaluation, scalability analysis

### Consensus - Multi-Model Perspectives

Consult multiple AI models with optional stance steering (for/against/neutral).

```bash
# Basic consensus with two models
uv run scripts/pal_consensus.py \
  --proposal "Should we use Redis for caching?" \
  --models gemini-2.5-flash gpt-4o

# With stance steering
uv run scripts/pal_consensus.py \
  --proposal "Microservices vs monolith?" \
  --models gemini-2.5-pro gpt-4o grok-3 \
  --stances for against neutral
```

**Options:**
- `--proposal` (required): The proposal or question
- `--models` (required): At least 2 models to consult
- `--stances`: Optional stance for each model (for/against/neutral)
- `--files`: Context files

### Clink - CLI Bridge

Spawn other AI CLIs (Gemini, Claude, Codex) with role presets.

```bash
# Use Gemini CLI with planner role
uv run scripts/pal_clink.py \
  --cli gemini \
  --role planner \
  --prompt "Plan this feature implementation"

# Use Claude CLI as code reviewer
uv run scripts/pal_clink.py \
  --cli claude \
  --role codereviewer \
  --prompt "Review this code" \
  --files src/auth.py
```

**CLIs:** gemini, claude, codex
**Roles:** default, planner, codereviewer

## Conversation Continuation

All tools support multi-turn conversations:

1. First call returns `continuation_id` in the response
2. Pass `--continuation-id <uuid>` to continue the conversation
3. Context from previous turns is automatically included
4. Works across different tools (chat -> codereview -> thinkdeep)

**Example:**
```bash
# Start conversation
uv run scripts/pal_chat.py --prompt "Explain this code" --files main.py --json
# Returns: {"continuation_id": "abc-123-..."}

# Continue conversation
uv run scripts/pal_chat.py --prompt "What about error handling?" \
  --continuation-id abc-123-...
```

## Configuration

Edit `config/config.yaml` to configure:

```yaml
# API Keys (or use environment variables)
api_keys:
  gemini: ${GEMINI_API_KEY}
  openai: ${OPENAI_API_KEY}
  xai: ${XAI_API_KEY}
  openrouter: ${OPENROUTER_API_KEY}
  custom_url: ${CUSTOM_API_URL}  # For Ollama

# Defaults
defaults:
  model: "auto"
  temperature: 1.0
  thinking_mode: "medium"
  locale: ""  # Optional: "Korean", "Japanese", etc.

# Conversation memory
conversation:
  max_turns: 50
  timeout_hours: 3
  storage: "memory"  # or "sqlite" for persistence

# Model restrictions (optional)
restrictions:
  google_allowed_models: []
  openai_allowed_models: []
  xai_allowed_models: []
  openrouter_allowed_models: []
```

## Supported Models

### Auto Mode
When `model: auto`, PAL selects the best available model:
1. Gemini 2.5 Flash (if GEMINI_API_KEY set)
2. GPT-4o (if OPENAI_API_KEY set)
3. Grok-3 (if XAI_API_KEY set)
4. OpenRouter default (if OPENROUTER_API_KEY set)
5. Local model (if CUSTOM_API_URL set)

### Specific Models

**Gemini** (requires GEMINI_API_KEY):
- gemini-2.5-flash (fast, cost-effective)
- gemini-2.5-pro (higher quality)
- gemini-2.0-flash-thinking-exp (experimental thinking)

**OpenAI** (requires OPENAI_API_KEY):
- gpt-4o (latest, multimodal)
- gpt-4-turbo
- o1, o3, o4 (reasoning models)

**X.AI** (requires XAI_API_KEY):
- grok-3 (latest)
- grok-3-mini (faster)

**OpenRouter** (requires OPENROUTER_API_KEY):
- Any model in provider/model format
- Example: `anthropic/claude-3-opus`, `meta-llama/llama-3-70b`
- Access 50+ models from various providers

**Custom/Ollama** (requires CUSTOM_API_URL):
- Any model name (e.g., llama3.2, codellama)
- Point to local Ollama: `export CUSTOM_API_URL="http://localhost:11434"`

## File Structure

```
.claude/skills/pal/
├── SKILL.md                    # This file
├── config/
│   ├── config.yaml             # Main configuration
│   └── cli_clients/            # CLI bridge configs
│       ├── gemini.yaml
│       ├── claude.yaml
│       └── codex.yaml
├── prompts/                    # System prompts
│   ├── chat.md
│   ├── thinkdeep.md
│   ├── codereview.md
│   ├── consensus.md
│   ├── debug.md
│   ├── analyze.md
│   └── clink/
├── scripts/                    # Executable tools
│   ├── pal_chat.py
│   ├── pal_thinkdeep.py
│   ├── pal_codereview.py
│   ├── pal_consensus.py
│   ├── pal_debug.py
│   ├── pal_analyze.py
│   ├── pal_clink.py
│   └── lib/                    # Shared libraries
├── examples/
│   └── workflows.md
└── requirements.txt
```

## Tips

1. **Use uv run** - no need to install dependencies first
2. **Use continuation_id** for iterative discussions - it preserves full context
3. **Start with ThinkDeep** for architectural questions - it uses more reasoning
4. **Use Consensus** for high-stakes decisions - get multiple perspectives
5. **Specify focus areas** in CodeReview for targeted analysis
6. **Use Debug** for systematic bug investigation with hypothesis testing
7. **Check JSON output** with `--json` for programmatic use
8. **Cross-tool continuation** - start with chat, continue with codereview, finish with thinkdeep

## Example Workflows

### Code Review Workflow
```bash
# 1. Initial review
uv run scripts/pal_codereview.py --files src/auth.py --focus security --json

# 2. Deep dive on findings (use continuation_id from step 1)
uv run scripts/pal_thinkdeep.py \
  --prompt "Analyze the SQL injection risk in detail" \
  --continuation-id <uuid>
```

### Debugging Workflow
```bash
# 1. Initial debug analysis
uv run scripts/pal_debug.py \
  --issue "Login fails with 500 error" \
  --files src/auth.py src/api.py \
  --error-logs "$(cat error.log)" --json

# 2. Get consensus on fix approach
uv run scripts/pal_consensus.py \
  --proposal "Should we add retry logic or fix the root cause?" \
  --models gemini-2.5-flash gpt-4o \
  --files src/auth.py
```

### Architecture Decision Workflow
```bash
# 1. Analyze current state
uv run scripts/pal_analyze.py \
  --prompt "Is our current architecture ready for 10x scale?" \
  --files src/

# 2. Get diverse perspectives
uv run scripts/pal_consensus.py \
  --proposal "Should we migrate to microservices?" \
  --models gemini-2.5-pro gpt-4o grok-3 \
  --stances for against neutral
```
