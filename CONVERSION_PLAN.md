# PAL MCP Server to Claude Code Skill Conversion Plan

## Executive Summary

This document outlines a comprehensive plan to convert the PAL MCP Server (Provider Abstraction Layer) into Claude Code Skills. The goal is to transform the 18 specialized AI tools currently implemented as an MCP server into modular skills that extend Claude Code's capabilities natively.

---

## Part 1: How the PAL MCP Server Works

### Architecture Overview

PAL MCP Server is a Model Context Protocol server that provides unified access to multiple AI providers (Gemini, OpenAI, XAI, OpenRouter, DIAL, Azure, Ollama) through specialized tools.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Claude Code (MCP Client)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    server.py (MCP Protocol)                      │
│  - Registers 18 tools                                           │
│  - Manages logging (mcp_server.log, mcp_activity.log)          │
│  - Validates API keys and configures providers                  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Simple Tools   │ │ Workflow Tools  │ │  Utility Tools  │
│  - chat         │ │ - codereview    │ │ - clink         │
│  - apilookup    │ │ - debug         │ │ - challenge     │
│                 │ │ - thinkdeep     │ │ - listmodels    │
│                 │ │ - secaudit      │ │ - version       │
│                 │ │ - refactor      │
│                 │ │ - testgen       │
│                 │ │ - docgen        │
│                 │ │ - planner       │
│                 │ │ - consensus     │
│                 │ │ - precommit     │
│                 │ │ - analyze       │
│                 │ │ - tracer        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              ModelProviderRegistry (Provider Selection)          │
│  Priority: GOOGLE → OPENAI → AZURE → XAI → DIAL → CUSTOM → OR  │
└─────────────────────────────────────────────────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
┌──────────┐           ┌──────────┐             ┌──────────┐
│  Gemini  │           │  OpenAI  │             │ Ollama   │
│  Provider│           │  Provider│             │ Provider │
└──────────┘           └──────────┘             └──────────┘
```

### Key Components

#### 1. Tool Framework (tools/)
- **BaseTool**: Core infrastructure for all tools
- **SimpleTool**: Single-turn request/response tools
- **WorkflowTool**: Multi-step investigation tools with forced pauses

#### 2. Provider Abstraction (providers/)
- **ModelProviderRegistry**: Central provider management
- **Providers**: Gemini, OpenAI, Azure, XAI, DIAL, OpenRouter, Custom/Ollama
- **Model Configs**: JSON files defining model capabilities (conf/*.json)

#### 3. Conversation Memory (utils/conversation_memory.py)
- UUID-based conversation threading
- Cross-tool context continuation
- File deduplication (newest-first prioritization)
- 50 turns max, 3-hour TTL

#### 4. System Prompts (systemprompts/)
- Tool-specific instruction sets
- CLI client prompts for clink tool

### Tool Inventory (18 Total)

| Tool | Type | Purpose |
|------|------|---------|
| chat | Simple | General development chat and code generation |
| apilookup | Simple | Quick web/API lookup instructions |
| thinkdeep | Workflow | Step-by-step deep thinking analysis |
| codereview | Workflow | Multi-pass code review (quality, security, performance) |
| debug | Workflow | Root cause analysis with hypothesis tracking |
| secaudit | Workflow | OWASP Top 10 security audit |
| docgen | Workflow | Documentation generation |
| refactor | Workflow | Code refactoring with severity levels |
| tracer | Workflow | Static call path analysis |
| testgen | Workflow | Test generation with expert validation |
| planner | Workflow | Interactive sequential planning |
| consensus | Workflow | Multi-model consensus analysis |
| precommit | Workflow | Pre-commit validation |
| analyze | Workflow | General file/code analysis |
| clink | Utility | CLI-to-CLI bridge for subagents |
| challenge | Utility | Critical challenge prompt |
| listmodels | Utility | List available AI models |
| version | Utility | Server version info |

---

## Part 2: How Claude Code Skills Work

### Skills vs MCP Servers

| Aspect | MCP Server | Claude Code Skill |
|--------|------------|-------------------|
| **Type** | Running process/executable | Markdown + YAML documentation |
| **Purpose** | External tool access | Teach Claude what to do |
| **Invocation** | Explicit tool calls | Automatic/contextual |
| **Token Cost** | Thousands+ per call | Dozens (metadata) to <5k |
| **Setup** | Technical (install, config) | Simple (drop files) |
| **External APIs** | Yes (providers) | No (uses Claude's knowledge) |

### Skill Structure

```
skill-name/
├── SKILL.md              # Required: Instructions + YAML frontmatter
├── reference.md          # Optional: Reference documentation
├── examples.md           # Optional: Code examples
├── scripts/              # Optional: Utility scripts
│   └── *.py, *.sh
└── templates/            # Optional: Templates
    └── *.md, *.json
```

### SKILL.md Format

```yaml
---
name: skill-name          # lowercase, hyphens, max 64 chars
description: "What it does and when to use it (max 1024 chars)"
allowed-tools: [read, write, bash]  # Optional tool restrictions
---

# Skill Instructions

Detailed instructions for how Claude should perform this task.
```

### Key Characteristics

1. **Model-invoked**: Claude determines when to use based on description
2. **Progressive disclosure**: Metadata first, full content on-demand
3. **Context-driven**: Activates automatically when description matches task
4. **No external APIs**: Uses Claude's existing capabilities

---

## Part 3: Conversion Strategy

### Fundamental Approach

The conversion requires a **paradigm shift**:

- **MCP Server**: Calls external AI models (Gemini, OpenAI, etc.) for analysis
- **Skills**: Guide Claude to perform the analysis itself using its own capabilities

This means:
- **External model calls → Claude's native intelligence**
- **Provider abstraction → Not needed (Claude is the model)**
- **Conversation memory → Claude's native context**
- **System prompts → Skill instructions**

### What Converts and What Doesn't

| Component | Conversion Approach |
|-----------|-------------------|
| System prompts | → Skill instructions (direct mapping) |
| Tool schemas | → Skill description + parameter guidance |
| Conversation memory | → Not needed (Claude has native context) |
| Provider registry | → Not applicable (Claude is the provider) |
| File handling | → Use Claude Code's native Read/Glob/Grep tools |
| Multi-model consensus | → Requires special handling (see below) |

### Skills to Create

Based on tool functionality and value-add potential:

#### High-Value Skills (Should Convert)

1. **codereview** - Multi-pass code review methodology
2. **debug** - Root cause analysis with hypothesis tracking
3. **secaudit** - OWASP-based security audit
4. **refactor** - Refactoring analysis with code smell detection
5. **testgen** - Test generation with coverage analysis
6. **thinkdeep** - Deep systematic thinking methodology
7. **docgen** - Documentation generation
8. **tracer** - Call path and dependency analysis
9. **planner** - Interactive planning methodology
10. **precommit** - Pre-commit validation checklist

#### Low-Value Skills (Skip or Simplify)

1. **chat** - Claude already does this natively
2. **apilookup** - Claude can use WebSearch/WebFetch
3. **analyze** - Too generic, covered by other skills
4. **challenge** - Simple prompt technique (include in others)
5. **listmodels** - Not applicable (no external providers)
6. **version** - Not applicable

#### Special Handling Required

1. **consensus** - Multi-model consensus requires external AI calls
   - Option A: Convert to "multiple perspectives" skill (Claude role-plays)
   - Option B: Keep as separate MCP tool for true multi-model
   - Option C: Use Claude's Task tool to spawn parallel analyses

2. **clink** - CLI-to-CLI bridge for subagents
   - Convert to instructions on using Task tool for subagent spawning

---

## Part 4: Detailed Conversion Plan

### Phase 1: Core Workflow Skills

#### 1.1 Code Review Skill

**Source**: `tools/workflow/codereview/`, `systemprompts/codereview_prompt.py`

```
.claude/skills/code-review/
├── SKILL.md           # Multi-pass review methodology
├── checklist.md       # Review checklist (quality, security, performance)
├── examples.md        # Sample review outputs
└── patterns.md        # Common anti-patterns to detect
```

**Key Instructions to Include**:
- 4-pass review process (quality, security, performance, architecture)
- Severity classification (critical, major, minor, info)
- Code snippet formatting for issues
- Positive feedback balance

#### 1.2 Debug Skill

**Source**: `tools/workflow/debug/`, `systemprompts/debug_prompt.py`

```
.claude/skills/debug/
├── SKILL.md           # Root cause analysis methodology
├── hypotheses.md      # Hypothesis tracking framework
├── techniques.md      # Debugging techniques by error type
└── examples.md        # Sample debug sessions
```

**Key Instructions to Include**:
- Hypothesis generation and ranking
- Evidence collection and verification
- Root cause vs symptom distinction
- Fix validation approach

#### 1.3 Security Audit Skill

**Source**: `tools/workflow/secaudit/`, `systemprompts/secaudit_prompt.py`

```
.claude/skills/security-audit/
├── SKILL.md           # Security audit methodology
├── owasp-top10.md     # OWASP Top 10 checklist
├── patterns.md        # Vulnerability patterns by language
└── remediation.md     # Remediation guidance
```

**Key Instructions to Include**:
- OWASP Top 10 coverage
- Input validation analysis
- Authentication/authorization checks
- Secrets detection
- SQL injection, XSS, CSRF patterns

#### 1.4 Refactor Skill

**Source**: `tools/workflow/refactor/`, `systemprompts/refactor_prompt.py`

```
.claude/skills/refactor/
├── SKILL.md           # Refactoring methodology
├── code-smells.md     # Code smell catalog
├── patterns.md        # Refactoring patterns
└── examples.md        # Before/after examples
```

**Key Instructions to Include**:
- Code smell detection (DRY, SOLID violations)
- Severity classification
- Incremental refactoring approach
- Test coverage considerations

#### 1.5 Test Generation Skill

**Source**: `tools/workflow/testgen/`, `systemprompts/testgen_prompt.py`

```
.claude/skills/test-generation/
├── SKILL.md           # Test generation methodology
├── patterns.md        # Test patterns by type (unit, integration)
├── coverage.md        # Coverage analysis approach
└── frameworks.md      # Framework-specific guidance
```

**Key Instructions to Include**:
- Happy path and edge case identification
- Mocking strategies
- Assertion patterns
- Test naming conventions
- Coverage target guidance

### Phase 2: Thinking and Analysis Skills

#### 2.1 Think Deep Skill

**Source**: `tools/workflow/thinkdeep/`, `systemprompts/thinkdeep_prompt.py`

```
.claude/skills/think-deep/
├── SKILL.md           # Deep thinking methodology
├── frameworks.md      # Thinking frameworks (5 whys, etc.)
└── examples.md        # Sample deep analyses
```

**Key Instructions to Include**:
- Step-by-step systematic thinking
- Multiple perspective analysis
- Assumption challenging
- Trade-off evaluation

#### 2.2 Documentation Generation Skill

**Source**: `tools/workflow/docgen/`, `systemprompts/docgen_prompt.py`

```
.claude/skills/doc-generation/
├── SKILL.md           # Documentation methodology
├── templates/
│   ├── function.md    # Function documentation template
│   ├── class.md       # Class documentation template
│   └── module.md      # Module documentation template
└── examples.md        # Sample documentation
```

**Key Instructions to Include**:
- Documentation style detection
- API documentation patterns
- README generation
- JSDoc/docstring conventions

#### 2.3 Call Tracer Skill

**Source**: `tools/workflow/tracer/`, `systemprompts/tracer_prompt.py`

```
.claude/skills/call-tracer/
├── SKILL.md           # Call path analysis methodology
├── patterns.md        # Dependency patterns
└── visualization.md   # ASCII diagram conventions
```

**Key Instructions to Include**:
- Static analysis approach
- Dependency graph construction
- Circular dependency detection
- Impact analysis

### Phase 3: Planning and Validation Skills

#### 3.1 Planner Skill

**Source**: `tools/workflow/planner/`, `systemprompts/planner_prompt.py`

```
.claude/skills/planner/
├── SKILL.md           # Planning methodology
├── templates/
│   ├── feature.md     # Feature planning template
│   └── refactor.md    # Refactor planning template
└── examples.md        # Sample plans
```

**Key Instructions to Include**:
- Task decomposition
- Dependency identification
- Risk assessment
- Milestone definition

#### 3.2 Pre-commit Skill

**Source**: `tools/workflow/precommit/`, `systemprompts/precommit_prompt.py`

```
.claude/skills/pre-commit/
├── SKILL.md           # Pre-commit validation methodology
├── checklist.md       # Pre-commit checklist
└── examples.md        # Sample validations
```

**Key Instructions to Include**:
- Code quality checks
- Test coverage verification
- Documentation updates
- Breaking change detection
- Commit message guidance

### Phase 4: Special Skills

#### 4.1 Multi-Perspective Analysis Skill (Replaces Consensus)

Since we can't call external models, convert consensus to multiple perspective analysis:

```
.claude/skills/multi-perspective/
├── SKILL.md           # Multi-perspective methodology
├── perspectives.md    # Common perspectives (devil's advocate, etc.)
└── synthesis.md       # Synthesis framework
```

**Key Instructions to Include**:
- Role-play different expert perspectives
- Advocate for/against positions
- Synthesize conflicting viewpoints
- Final recommendation with confidence

#### 4.2 Subagent Coordination Skill (Replaces clink)

Guide users on using Claude Code's Task tool:

```
.claude/skills/subagent-coordination/
├── SKILL.md           # Subagent usage methodology
└── patterns.md        # Common subagent patterns
```

**Key Instructions to Include**:
- When to use Task tool
- Subagent type selection
- Parallel vs sequential execution
- Result synthesis

---

## Part 5: Implementation Details

### Directory Structure

```
.claude/skills/
├── code-review/
│   ├── SKILL.md
│   ├── checklist.md
│   ├── examples.md
│   └── patterns.md
├── debug/
│   ├── SKILL.md
│   ├── hypotheses.md
│   ├── techniques.md
│   └── examples.md
├── security-audit/
│   ├── SKILL.md
│   ├── owasp-top10.md
│   ├── patterns.md
│   └── remediation.md
├── refactor/
│   ├── SKILL.md
│   ├── code-smells.md
│   ├── patterns.md
│   └── examples.md
├── test-generation/
│   ├── SKILL.md
│   ├── patterns.md
│   ├── coverage.md
│   └── frameworks.md
├── think-deep/
│   ├── SKILL.md
│   ├── frameworks.md
│   └── examples.md
├── doc-generation/
│   ├── SKILL.md
│   ├── templates/
│   └── examples.md
├── call-tracer/
│   ├── SKILL.md
│   ├── patterns.md
│   └── visualization.md
├── planner/
│   ├── SKILL.md
│   ├── templates/
│   └── examples.md
├── pre-commit/
│   ├── SKILL.md
│   ├── checklist.md
│   └── examples.md
├── multi-perspective/
│   ├── SKILL.md
│   ├── perspectives.md
│   └── synthesis.md
└── subagent-coordination/
    ├── SKILL.md
    └── patterns.md
```

### SKILL.md Template

```yaml
---
name: <skill-name>
description: "<Comprehensive description that includes trigger words. Max 1024 chars.>"
allowed-tools: [Read, Glob, Grep, Edit, Write, Bash, Task]
---

# <Skill Name>

## Overview
Brief description of what this skill does and when it activates.

## When to Use
- Trigger condition 1
- Trigger condition 2

## Methodology

### Step 1: [Name]
Detailed instructions...

### Step 2: [Name]
Detailed instructions...

## Output Format
How to structure the response...

## Examples
Reference to examples.md if needed...

## Best Practices
- Practice 1
- Practice 2
```

### Skill Description Guidelines

For automatic activation, descriptions should include:
- Action verbs users would use
- Problem types addressed
- File/code types handled
- Common trigger phrases

Example for code-review:
```yaml
description: "Conduct thorough multi-pass code reviews examining quality, security, performance, and architecture. Use when reviewing pull requests, analyzing code changes, assessing code quality, or evaluating new implementations. Includes SOLID principles, security vulnerability detection, and performance optimization analysis."
```

---

## Part 6: Migration Process

### Step 1: Extract System Prompts

For each tool, extract the system prompt from `systemprompts/<tool>_prompt.py`:

```python
# Example extraction
from systemprompts.codereview_prompt import get_system_prompt
prompt = get_system_prompt()
# Convert to skill instructions
```

### Step 2: Create SKILL.md Files

Transform system prompts into skill format:
- Remove API-specific instructions
- Remove provider/model references
- Add trigger descriptions
- Structure as methodology steps

### Step 3: Extract Reference Content

From tool implementations (`tools/workflow/<tool>/`):
- Extract checklists
- Extract patterns
- Extract examples
- Create supporting .md files

### Step 4: Test Skills

1. Place skills in `.claude/skills/` directory
2. Test activation with sample prompts
3. Verify instructions are followed
4. Iterate on descriptions for better activation

### Step 5: Documentation

Create README.md for the skills collection explaining:
- Available skills
- How to use each
- Trigger phrases
- Customization options

---

## Part 7: Limitations and Trade-offs

### What's Lost

1. **Multi-model capabilities**: Cannot call Gemini/OpenAI directly
2. **Provider flexibility**: Single model (Claude) only
3. **Conversation memory persistence**: Skills don't persist memory across sessions
4. **Token-level control**: Can't configure thinking budgets per-call
5. **True consensus**: Multiple model perspectives not available

### What's Gained

1. **Simplicity**: No MCP server setup required
2. **Integration**: Native Claude Code experience
3. **Performance**: No external API latency
4. **Cost**: No additional API costs
5. **Reliability**: No external service dependencies
6. **Portability**: Skills travel with project (`.claude/skills/`)

### Hybrid Approach

For users who need multi-model capabilities:
- Keep PAL MCP Server for consensus/multi-model tools
- Use skills for single-model methodologies
- Document when to use each

---

## Part 8: Implementation Timeline

### Phase 1: Core Skills (Priority)
1. code-review skill
2. debug skill
3. security-audit skill
4. refactor skill
5. test-generation skill

### Phase 2: Thinking Skills
6. think-deep skill
7. doc-generation skill
8. call-tracer skill

### Phase 3: Planning Skills
9. planner skill
10. pre-commit skill

### Phase 4: Special Skills
11. multi-perspective skill
12. subagent-coordination skill

### Phase 5: Testing and Documentation
13. Test all skills
14. Create usage documentation
15. Create migration guide from MCP

---

## Part 9: Success Criteria

1. All 12 skills activate correctly on relevant prompts
2. Skill outputs match quality of MCP tool outputs
3. Documentation is complete
4. Skills work without MCP server running
5. User feedback positive on usability

---

## Appendix A: System Prompt Locations

| Tool | System Prompt File |
|------|-------------------|
| chat | systemprompts/chat_prompt.py |
| codereview | systemprompts/codereview_prompt.py |
| debug | systemprompts/debug_prompt.py |
| secaudit | systemprompts/secaudit_prompt.py |
| refactor | systemprompts/refactor_prompt.py |
| testgen | systemprompts/testgen_prompt.py |
| thinkdeep | systemprompts/thinkdeep_prompt.py |
| docgen | systemprompts/docgen_prompt.py |
| tracer | systemprompts/tracer_prompt.py |
| planner | systemprompts/planner_prompt.py |
| precommit | systemprompts/precommit_prompt.py |
| analyze | systemprompts/analyze_prompt.py |
| consensus | systemprompts/consensus_prompt.py |

## Appendix B: Example Skill Implementation

See `.claude/skills/code-review/SKILL.md` for a complete example implementation.
