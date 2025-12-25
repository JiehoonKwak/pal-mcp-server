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

# Install dependencies
pip install -r requirements.txt

# Test
python scripts/pal_chat.py --prompt "Hello, PAL!" --json
```

## Available Tools

All tools are in the `scripts/` directory. Run them via Bash:

### Chat - Collaborative Thinking

General development chat with multi-turn conversation support.

```bash
python scripts/pal_chat.py \
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
python scripts/pal_thinkdeep.py \
  --prompt "Complex architectural question" \
  --files src/core.py \
  --thinking-mode high
```

**Defaults to `--thinking-mode high`** for deeper analysis.

### CodeReview - Expert Code Review

Comprehensive multi-pass code review with severity ratings.

```bash
python scripts/pal_codereview.py \
  --files src/main.py src/utils.py \
  --focus security performance \
  --model gemini-2.5-pro
```

**Focus areas:** security, performance, maintainability, architecture, testing

### Clink - CLI Bridge

Spawn other AI CLIs (Gemini, Claude, Codex) with role presets.

```bash
# Use Gemini CLI with planner role
python scripts/pal_clink.py \
  --cli gemini \
  --role planner \
  --prompt "Plan this feature implementation"

# Use Claude CLI as code reviewer
python scripts/pal_clink.py \
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
4. Works across different tools (chat → codereview → thinkdeep)

**Example:**
```bash
# Start conversation
python scripts/pal_chat.py --prompt "Explain this code" --files main.py --json
# Returns: {"continuation_id": "abc-123-..."}

# Continue conversation
python scripts/pal_chat.py --prompt "What about error handling?" \
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
  custom_url: ${CUSTOM_API_URL}  # For Ollama

# Defaults
defaults:
  model: "auto"
  temperature: 1.0
  thinking_mode: "medium"

# Conversation memory
conversation:
  max_turns: 50
  timeout_hours: 3
  storage: "memory"  # or "sqlite" for persistence
```

## When to Use PAL

- **Chat**: Brainstorming, second opinions, general development discussion
- **ThinkDeep**: Complex architectural decisions, deep analysis requiring extended thinking
- **CodeReview**: Comprehensive code review before commits/PRs with severity ratings
- **Clink**: Leveraging specific CLI capabilities (Gemini's web search, Claude's tooling)

## Supported Models

### Auto Mode
When `model: auto`, PAL selects the best available model:
1. Gemini 2.5 Flash (if GEMINI_API_KEY set)
2. GPT-4o (if OPENAI_API_KEY set)
3. Grok-3 (if XAI_API_KEY set)
4. Local model (if CUSTOM_API_URL set)

### Specific Models
- **Gemini**: gemini-2.5-flash, gemini-2.5-pro, gemini-3.0-pro
- **OpenAI**: gpt-4o, gpt-4-turbo, o1, o3, o4
- **X.AI**: grok-3, grok-3-mini
- **OpenRouter**: any-provider/model-name (e.g., anthropic/claude-3-opus)
- **Custom/Ollama**: Any model name (uses CUSTOM_API_URL)

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
│   └── clink/
├── scripts/                    # Executable tools
│   ├── pal_chat.py
│   ├── pal_thinkdeep.py
│   ├── pal_codereview.py
│   ├── pal_clink.py
│   └── lib/                    # Shared libraries
└── requirements.txt
```

## Tips

1. **Use continuation_id** for iterative discussions - it preserves full context
2. **Start with ThinkDeep** for architectural questions - it uses more reasoning
3. **Specify focus areas** in CodeReview for targeted analysis
4. **Use Clink** when you need specific CLI features (web search, etc.)
5. **Check JSON output** with `--json` for programmatic use
