# PAL MCP Server to Claude Code Skill Conversion Plan (Enhanced)

## Executive Summary

This plan converts the PAL MCP Server into a **portable, configurable Claude Code skill collection** that preserves ALL functionality including external API calls, multi-turn conversations, and CLI bridging through Python scripts.

---

## Part 1: How This MCP Server Works

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PAL MCP Server Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐   │
│  │ Claude Code │────▶│   server.py      │────▶│   Provider Registry     │   │
│  │ (MCP Client)│     │ (MCP Protocol)   │     │ (Gemini/OpenAI/XAI/...) │   │
│  └─────────────┘     └──────────────────┘     └─────────────────────────┘   │
│         │                    │                           │                   │
│         │                    ▼                           ▼                   │
│         │           ┌────────────────────────────────────────────────┐      │
│         │           │              Tool Layer (18 tools)              │      │
│         │           ├────────────────────────────────────────────────┤      │
│         │           │ Simple Tools    │ Workflow Tools │ Utility     │      │
│         │           │ • chat          │ • codereview   │ • clink     │      │
│         │           │ • apilookup     │ • debug        │ • challenge │      │
│         │           │                 │ • thinkdeep    │ • listmodels│      │
│         │           │                 │ • secaudit     │ • version   │      │
│         │           │                 │ • refactor     │             │      │
│         │           │                 │ • testgen      │             │      │
│         │           │                 │ • docgen       │             │      │
│         │           │                 │ • tracer       │             │      │
│         │           │                 │ • planner      │             │      │
│         │           │                 │ • consensus    │             │      │
│         │           │                 │ • precommit    │             │      │
│         │           │                 │ • analyze      │             │      │
│         │           └────────────────────────────────────────────────┘      │
│         │                              │                                     │
│         │                              ▼                                     │
│         │           ┌────────────────────────────────────────────────┐      │
│         │           │         Conversation Memory System              │      │
│         │           │ • UUID-based threading                          │      │
│         │           │ • Cross-tool continuation                       │      │
│         │           │ • File deduplication (newest-first)            │      │
│         │           │ • 50 turns max, 3-hour TTL                     │      │
│         │           └────────────────────────────────────────────────┘      │
│         │                              │                                     │
│         │                              ▼                                     │
│         │           ┌────────────────────────────────────────────────┐      │
│         └──────────▶│           clink (CLI Bridge)                    │      │
│                     │ • Spawns Gemini CLI, Claude CLI, Codex CLI     │      │
│                     │ • Role-based prompts (planner, codereviewer)   │      │
│                     │ • Full context passing                          │      │
│                     └────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components Explained

#### 1. Conversation Memory (`utils/conversation_memory.py`)
- **UUID-based threads** for multi-turn conversations
- **Cross-tool continuation** - switch from chat → codereview → debug while maintaining context
- **Newest-first file prioritization** - when same file appears in multiple turns, newest wins
- **Token-aware** - respects model limits during context reconstruction

#### 2. Chat Tool (`tools/chat.py`)
- General development chat with brainstorming
- Supports file context and images
- Code generation capability with artifact saving
- Multi-turn conversation via `continuation_id`

#### 3. ThinkDeep Tool (Workflow)
- Step-by-step deep analysis
- Configurable thinking modes (minimal → max)
- Extended thinking budget support
- Multi-turn for iterative reasoning

#### 4. Clink Tool (`tools/clink.py`)
- **CLI-to-CLI bridge** for spawning other AI CLIs
- Supports: Gemini CLI, Claude CLI, Codex CLI
- Role presets: default, planner, codereviewer
- Configured via `conf/cli_clients/*.json`

---

## Part 2: Why Functionality is NOT Lost

### Skills Can Execute Scripts!

Claude Code skills support a `scripts/` directory that can contain Python scripts. These scripts can:

1. **Make external API calls** to Gemini, OpenAI, XAI, etc.
2. **Manage conversation state** via files or SQLite
3. **Spawn CLI subprocesses** like Gemini CLI
4. **Handle complex logic** that pure instructions can't express

### The Skill + Script Architecture

```
pal-skills/
├── SKILL.md                    # Entry point - activates on relevant tasks
├── config/
│   ├── config.yaml             # User configuration (API keys, defaults)
│   ├── providers.yaml          # Provider definitions
│   └── cli_clients/            # CLI bridge configurations
│       ├── gemini.yaml
│       ├── claude.yaml
│       └── codex.yaml
├── prompts/                    # Organized system prompts
│   ├── chat.md
│   ├── thinkdeep.md
│   ├── codereview.md
│   └── ...
├── scripts/                    # Python scripts for execution
│   ├── pal_chat.py            # Chat with external models
│   ├── pal_thinkdeep.py       # Deep thinking with external models
│   ├── pal_codereview.py      # Code review with external models
│   ├── pal_clink.py           # CLI bridge (spawn Gemini/Claude CLI)
│   ├── pal_consensus.py       # Multi-model consensus
│   └── lib/                   # Shared libraries
│       ├── __init__.py
│       ├── providers.py       # API client wrappers
│       ├── conversation.py    # Conversation memory
│       ├── config.py          # Configuration loader
│       └── utils.py           # Shared utilities
└── examples/                   # Usage examples
    └── workflows.md
```

---

## Part 3: Detailed Conversion Strategy

### What Each Component Becomes

| MCP Component | Skill Equivalent |
|---------------|------------------|
| `server.py` | `SKILL.md` (instructions) + `scripts/` (execution) |
| `providers/*.py` | `scripts/lib/providers.py` |
| `utils/conversation_memory.py` | `scripts/lib/conversation.py` |
| `systemprompts/*.py` | `prompts/*.md` (organized markdown) |
| `conf/cli_clients/*.json` | `config/cli_clients/*.yaml` |
| `tools/*.py` | Individual `scripts/pal_*.py` + `prompts/*.md` |

### Configuration Approach

**config/config.yaml** (User-editable):
```yaml
# PAL Skills Configuration
version: "1.0"

# API Keys (can also use environment variables)
api_keys:
  gemini: ${GEMINI_API_KEY}
  openai: ${OPENAI_API_KEY}
  xai: ${XAI_API_KEY}
  openrouter: ${OPENROUTER_API_KEY}
  custom_url: ${CUSTOM_API_URL}

# Default settings
defaults:
  model: "auto"  # or specific model like "gemini-2.5-flash"
  temperature: 1.0
  thinking_mode: "medium"  # minimal, low, medium, high, max
  locale: ""  # e.g., "Korean" for localized responses

# Conversation memory
conversation:
  max_turns: 50
  timeout_hours: 3
  storage: "memory"  # or "sqlite" for persistence

# Model restrictions (optional)
restrictions:
  google_allowed_models: []  # empty = all allowed
  openai_allowed_models: []

# CLI clients for clink
cli_clients:
  default: "gemini"
  enabled:
    - gemini
    - claude
    - codex
```

### Priority Tools (Chat, ThinkDeep, Clink)

#### 1. Chat Skill with Multi-Turn

**prompts/chat.md**:
```markdown
# Chat System Prompt

You are a senior engineering thought-partner collaborating with another AI agent.
Your mission is to brainstorm, validate ideas, and offer well-reasoned second opinions.

## Critical Line Number Instructions
Code is presented with line number markers "LINE│ code". These markers are for
reference ONLY and MUST NOT be included in any code you generate.

## If More Information Needed
If discussing specific code without context, respond ONLY with:
{
  "status": "files_required_to_continue",
  "mandatory_instructions": "<your instructions>",
  "files_needed": ["[file path]"]
}

## Scope & Focus
- Ground suggestions in the project's current tech stack
- Avoid over-engineered solutions
- Keep proposals practical and actionable

## Collaboration Approach
1. Treat the collaborating agent as an equally senior peer
2. Engage deeply - extend, refine, explore alternatives when justified
3. Examine edge cases and failure modes
4. Challenge assumptions constructively
5. Provide concrete examples and actionable next steps
```

**scripts/pal_chat.py**:
```python
#!/usr/bin/env python3
"""
PAL Chat - Multi-turn conversational chat with external models.

Usage:
    python pal_chat.py --prompt "Your question" [options]

Options:
    --files FILE [FILE ...]    Files to include as context
    --model MODEL              Model to use (default: from config)
    --continuation-id ID       Continue existing conversation
    --working-dir PATH         Working directory for artifacts
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from config import load_config
from providers import get_provider, execute_request
from conversation import ConversationMemory

def main():
    parser = argparse.ArgumentParser(description="PAL Chat")
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--files", nargs="*", default=[], help="Files to include")
    parser.add_argument("--images", nargs="*", default=[], help="Images to include")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--continuation-id", help="Continue conversation")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--thinking-mode", choices=["minimal", "low", "medium", "high", "max"])

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Get or create conversation thread
    memory = ConversationMemory(config)

    if args.continuation_id:
        thread = memory.get_thread(args.continuation_id)
        conversation_history = memory.build_history(thread)
    else:
        thread = memory.create_thread("chat", {"prompt": args.prompt})
        conversation_history = ""

    # Load system prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "chat.md"
    system_prompt = prompt_path.read_text() if prompt_path.exists() else ""

    # Read files if provided
    file_content = ""
    if args.files:
        from utils import read_files
        file_content = read_files(args.files)

    # Build full prompt
    full_prompt = f"{system_prompt}\n\n"
    if conversation_history:
        full_prompt += f"{conversation_history}\n\n"
    if file_content:
        full_prompt += f"=== FILES ===\n{file_content}\n\n"
    full_prompt += f"=== USER REQUEST ===\n{args.prompt}"

    # Get provider and execute
    model = args.model or config.get("defaults", {}).get("model", "gemini-2.5-flash")
    provider = get_provider(model, config)

    response = execute_request(
        provider=provider,
        prompt=full_prompt,
        model=model,
        temperature=args.temperature or config.get("defaults", {}).get("temperature", 1.0),
        thinking_mode=args.thinking_mode or config.get("defaults", {}).get("thinking_mode", "medium"),
        images=args.images,
    )

    # Record turn for continuation
    memory.add_turn(
        thread_id=thread["thread_id"],
        role="user",
        content=args.prompt,
        files=args.files,
    )
    memory.add_turn(
        thread_id=thread["thread_id"],
        role="assistant",
        content=response["content"],
        model_name=model,
    )

    # Output result
    result = {
        "status": "success",
        "content": response["content"],
        "continuation_id": thread["thread_id"],
        "model": model,
        "usage": response.get("usage", {}),
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

#### 2. ThinkDeep Skill with Multi-Turn

**prompts/thinkdeep.md**:
```markdown
# ThinkDeep System Prompt

## Role
You are a senior engineering collaborator working alongside the agent on complex
software problems. The agent will send you content—analysis, prompts, questions,
ideas, or theories—to deepen, validate, or extend with rigor and clarity.

## Guidelines
1. Begin with context analysis: identify tech stack, languages, frameworks, constraints
2. Stay on scope: avoid speculative or oversized ideas
3. Challenge and enrich: find gaps, question assumptions, surface hidden complexities
4. Provide actionable next steps: specific advice, trade-offs, implementation strategies
5. Use concise, technical language for experienced engineers

## Key Focus Areas
- Architecture & Design: modularity, boundaries, dependencies
- Performance & Scalability: efficiency, concurrency, bottlenecks
- Security & Safety: validation, auth, error handling, vulnerabilities
- Quality & Maintainability: readability, testing, monitoring
- Integration & Deployment: compatibility, configuration, operations

## Evaluation
Your response will be reviewed before decisions. Goal: extend thinking, surface
blind spots, refine options—not deliver final answers in isolation.
```

**scripts/pal_thinkdeep.py**:
```python
#!/usr/bin/env python3
"""
PAL ThinkDeep - Deep systematic analysis with extended thinking.

Usage:
    python pal_thinkdeep.py --prompt "Your question" [options]
"""
# Similar structure to chat, but uses thinkdeep.md prompt
# and defaults to higher thinking_mode
```

#### 3. Clink Skill (CLI Bridge)

**config/cli_clients/gemini.yaml**:
```yaml
name: gemini
command: gemini
additional_args:
  - "--yolo"
env: {}
roles:
  default:
    prompt_path: prompts/clink/default.md
    role_args: []
  planner:
    prompt_path: prompts/clink/planner.md
    role_args: []
  codereviewer:
    prompt_path: prompts/clink/codereviewer.md
    role_args: []
```

**scripts/pal_clink.py**:
```python
#!/usr/bin/env python3
"""
PAL Clink - CLI-to-CLI bridge for spawning AI CLIs.

Usage:
    python pal_clink.py --cli gemini --role planner --prompt "Plan this feature"
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from config import load_config, load_cli_client

def main():
    parser = argparse.ArgumentParser(description="PAL Clink - CLI Bridge")
    parser.add_argument("--prompt", required=True, help="Prompt to send")
    parser.add_argument("--cli", default="gemini", help="CLI to use")
    parser.add_argument("--role", default="default", help="Role preset")
    parser.add_argument("--files", nargs="*", default=[], help="Files to include")

    args = parser.parse_args()

    config = load_config()
    client = load_cli_client(args.cli)
    role = client["roles"].get(args.role, client["roles"]["default"])

    # Build command
    cmd = [client["command"]] + client.get("additional_args", [])

    # Load role prompt
    prompt_path = Path(__file__).parent.parent / role["prompt_path"]
    system_prompt = prompt_path.read_text() if prompt_path.exists() else ""

    # Build full prompt
    full_prompt = f"{system_prompt}\n\n=== USER REQUEST ===\n{args.prompt}"

    # Add files if provided
    if args.files:
        file_args = []
        for f in args.files:
            file_args.extend(["--file", f])
        cmd.extend(file_args)

    # Execute CLI
    cmd.extend(["--prompt", full_prompt])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env={**dict(os.environ), **client.get("env", {})},
        )

        output = {
            "status": "success" if result.returncode == 0 else "error",
            "content": result.stdout,
            "stderr": result.stderr,
            "cli": args.cli,
            "role": args.role,
        }

    except subprocess.TimeoutExpired:
        output = {
            "status": "error",
            "content": f"CLI '{args.cli}' timed out after 300 seconds",
        }
    except Exception as e:
        output = {
            "status": "error",
            "content": str(e),
        }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
```

---

## Part 4: Complete Folder Structure

```
.claude/skills/pal/
├── SKILL.md                           # Main skill entry point
│
├── config/                            # All configuration (portable)
│   ├── config.yaml                    # Main user configuration
│   ├── providers.yaml                 # Provider definitions
│   └── cli_clients/                   # CLI bridge configs
│       ├── gemini.yaml
│       ├── claude.yaml
│       └── codex.yaml
│
├── prompts/                           # Organized system prompts
│   ├── chat.md                        # Chat prompt
│   ├── thinkdeep.md                   # Deep thinking prompt
│   ├── codereview.md                  # Code review prompt
│   ├── debug.md                       # Debug prompt
│   ├── secaudit.md                    # Security audit prompt
│   ├── refactor.md                    # Refactoring prompt
│   ├── testgen.md                     # Test generation prompt
│   ├── docgen.md                      # Documentation prompt
│   ├── tracer.md                      # Call tracer prompt
│   ├── planner.md                     # Planning prompt
│   ├── consensus.md                   # Consensus prompt
│   ├── precommit.md                   # Pre-commit prompt
│   ├── analyze.md                     # Analysis prompt
│   ├── generate_code.md               # Code generation addon
│   └── clink/                         # CLI bridge role prompts
│       ├── default.md
│       ├── planner.md
│       └── codereviewer.md
│
├── scripts/                           # Executable Python scripts
│   ├── pal_chat.py                    # Chat tool
│   ├── pal_thinkdeep.py               # ThinkDeep tool
│   ├── pal_codereview.py              # Code review tool
│   ├── pal_debug.py                   # Debug tool
│   ├── pal_secaudit.py                # Security audit tool
│   ├── pal_refactor.py                # Refactor tool
│   ├── pal_testgen.py                 # Test generation tool
│   ├── pal_docgen.py                  # Documentation tool
│   ├── pal_tracer.py                  # Call tracer tool
│   ├── pal_planner.py                 # Planner tool
│   ├── pal_consensus.py               # Multi-model consensus
│   ├── pal_precommit.py               # Pre-commit validation
│   ├── pal_clink.py                   # CLI bridge tool
│   │
│   └── lib/                           # Shared library code
│       ├── __init__.py
│       ├── config.py                  # Configuration loading
│       ├── providers.py               # Provider implementations
│       │   ├── base.py                # Base provider class
│       │   ├── gemini.py              # Gemini API client
│       │   ├── openai.py              # OpenAI API client
│       │   ├── xai.py                 # XAI API client
│       │   ├── openrouter.py          # OpenRouter client
│       │   └── custom.py              # Custom/Ollama client
│       ├── conversation.py            # Conversation memory
│       ├── file_utils.py              # File handling
│       ├── token_utils.py             # Token estimation
│       └── utils.py                   # General utilities
│
├── examples/                          # Usage examples
│   ├── workflows.md                   # Common workflow examples
│   └── sample_config.yaml             # Sample configuration
│
└── requirements.txt                   # Python dependencies
```

---

## Part 5: Main SKILL.md Entry Point

```yaml
---
name: pal
description: "PAL (Provider Abstraction Layer) - Multi-model AI development assistant. Use for code review, debugging, security audits, refactoring, test generation, documentation, planning, and collaborative thinking. Supports conversation continuation, multi-model consensus, and CLI bridging to Gemini/Claude/Codex. Activate when user needs deep analysis, second opinions, or expert code review."
allowed-tools: [Read, Glob, Grep, Edit, Write, Bash, Task]
---

# PAL - Provider Abstraction Layer Skills

PAL provides access to multiple AI models (Gemini, OpenAI, XAI, OpenRouter, custom)
for specialized development tasks with conversation memory and cross-tool continuation.

## Available Tools

Execute tools via Bash with the scripts in this skill:

### Chat - Collaborative Thinking
```bash
python scripts/pal_chat.py \
  --prompt "Your question or discussion topic" \
  --files path/to/file.py \
  --model gemini-2.5-flash \
  --continuation-id <uuid-for-multi-turn>
```

### ThinkDeep - Deep Analysis
```bash
python scripts/pal_thinkdeep.py \
  --prompt "Complex problem to analyze" \
  --files path/to/code.py \
  --thinking-mode high
```

### Code Review
```bash
python scripts/pal_codereview.py \
  --files path/to/code.py \
  --focus "security,performance"
```

### Debug - Root Cause Analysis
```bash
python scripts/pal_debug.py \
  --prompt "Describe the bug" \
  --files path/to/buggy.py
```

### Security Audit
```bash
python scripts/pal_secaudit.py \
  --files path/to/code.py \
  --owasp-focus "injection,auth"
```

### Clink - CLI Bridge
```bash
python scripts/pal_clink.py \
  --cli gemini \
  --role planner \
  --prompt "Plan this feature implementation"
```

### Consensus - Multi-Model Analysis
```bash
python scripts/pal_consensus.py \
  --prompt "Should we use microservices or monolith?" \
  --models gemini-2.5-pro,gpt-4o,grok-3 \
  --stance for,against,neutral
```

## Configuration

Edit `config/config.yaml` to configure:
- API keys (or use environment variables)
- Default model and temperature
- Thinking mode preferences
- Conversation memory settings
- Enabled CLI clients

## Conversation Continuation

All tools support multi-turn conversations:

1. First call returns `continuation_id` in response
2. Pass `--continuation-id <uuid>` to continue the conversation
3. Context from previous turns is automatically included
4. Works across different tools (chat → codereview → debug)

## When to Use PAL

- **Chat**: Brainstorming, second opinions, general development discussion
- **ThinkDeep**: Complex architectural decisions, deep analysis
- **CodeReview**: Comprehensive code review before commits/PRs
- **Debug**: Finding root cause of bugs with hypothesis tracking
- **SecAudit**: Security vulnerability scanning (OWASP Top 10)
- **Consensus**: Getting multiple AI perspectives on decisions
- **Clink**: Leveraging specific CLI capabilities (Gemini's web search, etc.)
```

---

## Part 6: Configuration & Portability

### Installation (Portable)

1. Copy entire `pal/` folder to `.claude/skills/pal/`
2. Copy `config/sample_config.yaml` to `config/config.yaml`
3. Edit `config/config.yaml` with your API keys
4. Install dependencies: `pip install -r requirements.txt`

### Environment Variables (Alternative to config file)

```bash
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export XAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export CUSTOM_API_URL="http://localhost:11434"  # For Ollama
```

### Project-Level Customization

Override settings per-project by creating `.claude/skills/pal/config/config.local.yaml`:

```yaml
# Project-specific overrides
defaults:
  model: "gemini-2.5-flash"  # This project uses Gemini
  locale: "Korean"           # Responses in Korean

restrictions:
  # Only allow specific models for this project
  google_allowed_models:
    - gemini-2.5-flash
    - gemini-2.5-pro
```

---

## Part 7: Preserving Key Features

### 1. Multi-Turn Conversation Memory

**scripts/lib/conversation.py**:
```python
"""
Conversation memory with UUID-based threading and cross-tool continuation.
Matches MCP server's conversation_memory.py behavior exactly.
"""

import json
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

class ConversationMemory:
    """Persistent conversation memory with cross-tool continuation."""

    def __init__(self, config: dict):
        self.max_turns = config.get("conversation", {}).get("max_turns", 50)
        self.timeout_hours = config.get("conversation", {}).get("timeout_hours", 3)
        storage_type = config.get("conversation", {}).get("storage", "memory")

        if storage_type == "sqlite":
            self.storage = SQLiteStorage(Path.home() / ".pal" / "conversations.db")
        else:
            self.storage = InMemoryStorage()

    def create_thread(self, tool_name: str, initial_context: dict) -> dict:
        """Create new conversation thread, return thread_id."""
        thread_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        thread = {
            "thread_id": thread_id,
            "created_at": now,
            "last_updated_at": now,
            "tool_name": tool_name,
            "turns": [],
            "initial_context": initial_context,
        }

        self.storage.set(thread_id, thread, ttl=self.timeout_hours * 3600)
        return thread

    def get_thread(self, thread_id: str) -> Optional[dict]:
        """Get thread by ID."""
        return self.storage.get(thread_id)

    def add_turn(self, thread_id: str, role: str, content: str,
                 files: list = None, model_name: str = None) -> bool:
        """Add turn to thread."""
        thread = self.get_thread(thread_id)
        if not thread or len(thread["turns"]) >= self.max_turns:
            return False

        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "model_name": model_name,
        }

        thread["turns"].append(turn)
        thread["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        self.storage.set(thread_id, thread, ttl=self.timeout_hours * 3600)
        return True

    def build_history(self, thread: dict, max_tokens: int = 100000) -> str:
        """Build formatted conversation history with newest-first prioritization."""
        if not thread or not thread.get("turns"):
            return ""

        turns = thread["turns"]

        # Collect files from all turns (newest-first deduplication)
        seen_files = set()
        all_files = []
        for turn in reversed(turns):
            for f in (turn.get("files") or []):
                if f not in seen_files:
                    seen_files.add(f)
                    all_files.append(f)

        parts = [
            "=== CONVERSATION HISTORY (CONTINUATION) ===",
            f"Thread: {thread['thread_id']}",
            f"Tool: {thread['tool_name']}",
            f"Turn {len(turns)}/{self.max_turns}",
            "You are continuing this conversation from where it left off.",
            "",
        ]

        # Add previous turns
        parts.append("Previous conversation turns:")
        for i, turn in enumerate(turns, 1):
            role_label = "Agent" if turn["role"] == "user" else (turn.get("model_name") or "Assistant")
            parts.append(f"\n--- Turn {i} ({role_label}) ---")
            if turn.get("files"):
                parts.append(f"Files: {', '.join(turn['files'])}")
            parts.append(turn["content"])

        parts.extend([
            "",
            "=== END CONVERSATION HISTORY ===",
            "",
            "IMPORTANT: Continue this conversation. Build on previous exchanges.",
        ])

        return "\n".join(parts)
```

### 2. Provider Abstraction

**scripts/lib/providers.py**:
```python
"""
Multi-provider AI client abstraction.
Supports: Gemini, OpenAI, XAI, OpenRouter, Custom/Ollama
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

class BaseProvider(ABC):
    """Base class for all AI providers."""

    @abstractmethod
    def generate(self, prompt: str, model: str, **kwargs) -> dict:
        pass

class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, model: str, **kwargs) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 1.0),
        )

        # Add thinking mode if supported
        thinking_mode = kwargs.get("thinking_mode", "medium")
        if thinking_mode and "2.5" in model:
            budgets = {"minimal": 0.005, "low": 0.08, "medium": 0.33, "high": 0.67, "max": 1.0}
            max_thinking = 24576  # Gemini 2.5 default
            budget = int(max_thinking * budgets.get(thinking_mode, 0.33))
            config.thinking_config = types.ThinkingConfig(thinking_budget=budget)

        response = self.client.models.generate_content(
            model=model,
            contents=[{"parts": [{"text": prompt}]}],
            config=config,
        )

        return {
            "content": response.text,
            "usage": {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
        }

class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str, **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 1.0),
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        }

def get_provider(model: str, config: dict) -> BaseProvider:
    """Get appropriate provider for model."""
    api_keys = config.get("api_keys", {})

    # Resolve 'auto' mode
    if model.lower() == "auto":
        # Priority: Gemini > OpenAI > XAI > Custom
        if api_keys.get("gemini"):
            return GeminiProvider(api_keys["gemini"]), "gemini-2.5-flash"
        elif api_keys.get("openai"):
            return OpenAIProvider(api_keys["openai"]), "gpt-4o"
        # ... etc

    # Route by model name pattern
    if "gemini" in model.lower():
        return GeminiProvider(api_keys.get("gemini") or os.getenv("GEMINI_API_KEY"))
    elif "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
        return OpenAIProvider(api_keys.get("openai") or os.getenv("OPENAI_API_KEY"))
    # ... etc

def execute_request(provider: BaseProvider, prompt: str, model: str, **kwargs) -> dict:
    """Execute request with retry logic."""
    max_retries = 4
    delays = [1, 3, 5, 8]

    for attempt in range(max_retries):
        try:
            return provider.generate(prompt, model, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            import time
            time.sleep(delays[attempt])
```

---

## Part 8: Implementation Priority

### Phase 1: Core Infrastructure
1. `scripts/lib/config.py` - Configuration loading
2. `scripts/lib/providers.py` - Provider abstraction (Gemini, OpenAI)
3. `scripts/lib/conversation.py` - Conversation memory
4. `scripts/lib/file_utils.py` - File handling with line numbers
5. `SKILL.md` - Main skill entry point

### Phase 2: Priority Tools (User Requested)
6. `scripts/pal_chat.py` - Chat with multi-turn
7. `scripts/pal_thinkdeep.py` - ThinkDeep with multi-turn
8. `scripts/pal_clink.py` - CLI bridge

### Phase 3: Prompts
9. `prompts/chat.md`
10. `prompts/thinkdeep.md`
11. `prompts/clink/*.md`

### Phase 4: Additional Tools
12. `scripts/pal_codereview.py`
13. `scripts/pal_debug.py`
14. `scripts/pal_secaudit.py`
15. `scripts/pal_consensus.py`
16. `scripts/pal_refactor.py`
17. `scripts/pal_testgen.py`
18. Additional prompts

### Phase 5: Polish
19. `requirements.txt`
20. `examples/workflows.md`
21. Configuration validation
22. Error handling improvements

---

## Part 9: Preserved Capabilities

| Feature | How Preserved |
|---------|---------------|
| Multi-model API calls | Python scripts call Gemini/OpenAI/XAI APIs directly |
| Conversation memory | `scripts/lib/conversation.py` with SQLite persistence option |
| Cross-tool continuation | Same `continuation_id` works across all tools |
| File deduplication | Newest-first logic in `build_history()` |
| CLI bridging (clink) | `scripts/pal_clink.py` spawns CLI subprocesses |
| Multi-model consensus | `scripts/pal_consensus.py` queries multiple models |
| Configurable | `config/config.yaml` + env vars + local overrides |
| Portable | Copy folder to any project's `.claude/skills/` |
| Thinking modes | Passed through to provider API calls |
| Line numbers | `file_utils.py` adds LINE│ markers |
| Token management | Token estimation and budget tracking |

---

## Appendix: Quick Start

```bash
# 1. Copy skill to project
cp -r pal-skills ~/.claude/skills/pal

# 2. Configure
cp ~/.claude/skills/pal/config/sample_config.yaml ~/.claude/skills/pal/config/config.yaml
# Edit config.yaml with your API keys

# 3. Install dependencies
pip install google-genai openai httpx pyyaml

# 4. Test
python ~/.claude/skills/pal/scripts/pal_chat.py --prompt "Hello, PAL!"
```

The skill will automatically activate when Claude Code detects relevant tasks!
