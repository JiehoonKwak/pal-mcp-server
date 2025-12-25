# PAL Skills Conversion Plan

## Executive Summary

This document provides a comprehensive analysis of discrepancies between the PAL MCP Server and the Claude Skills implementation, along with a detailed plan to achieve feature parity.

---

## Part 1: Diagnosis

### 1.1 Architecture Overview

**MCP Server Architecture:**
```
Claude CLI ←→ MCP Server (server.py) ←→ External Models (Gemini/OpenAI/etc.)
                    ↓
              Tool Registry (18 tools)
                    ↓
              Providers (6 providers)
                    ↓
              Conversation Memory (in-memory, 3hr TTL)
```

**Skills Architecture:**
```
Claude CLI ←→ Skill Scripts (PEP 723) ←→ External Models
                    ↓
              lib/ modules
                    ↓
              File-based storage
```

### 1.2 Tool Coverage Analysis

| MCP Tool | Skill Script | Status | Gap Severity |
|----------|--------------|--------|--------------|
| chat | pal_chat.py | ✅ Exists | Low |
| thinkdeep | pal_thinkdeep.py | ✅ Exists | Medium |
| codereview | pal_codereview.py | ✅ Exists | Medium |
| consensus | pal_consensus.py | ⚠️ Partial | **Critical** |
| debug | pal_debug.py | ⚠️ Partial | **Critical** |
| analyze | pal_analyze.py | ⚠️ Partial | **Critical** |
| clink | pal_clink.py | ⚠️ Partial | High |
| planner | ❌ Missing | **Missing** | High |
| precommit | ❌ Missing | **Missing** | Medium |
| secaudit | ❌ Missing | **Missing** | Medium |
| docgen | ❌ Missing | **Missing** | Medium |
| refactor | ❌ Missing | **Missing** | Medium |
| tracer | ❌ Missing | **Missing** | Low |
| testgen | ❌ Missing | **Missing** | Medium |
| challenge | ❌ Missing | **Missing** | Low |
| apilookup | ❌ Missing | **Missing** | Low |
| listmodels | ❌ Missing | **Missing** | Low |
| version | ❌ Missing | **Missing** | Low |

---

### 1.3 Critical Discrepancies

#### 1.3.1 Consensus Tool - **BROKEN**

**MCP Implementation** (`tools/consensus.py`):
- Multi-model querying (queries 2+ models simultaneously)
- Stance steering with dynamic prompt injection (`for`/`against`/`neutral`)
- Response synthesis across multiple model perspectives
- Structured output with confidence scoring
- `{stance_prompt}` placeholder replaced at runtime

**Skill Implementation** (`pal_consensus.py`):
```python
# Current: Single model call, no stance support
response = await provider.generate(prompt, ...)
```

**Missing Features:**
1. `stance` parameter (`for`/`against`/`neutral`)
2. Multi-model parallel querying
3. Response synthesis logic
4. Dynamic stance prompt injection

**Impact:** Users cannot get multi-perspective consensus analysis

---

#### 1.3.2 Debug Tool - **BROKEN**

**MCP Implementation** (`tools/debug.py` - WorkflowTool subclass):
```python
class DebugInvestigationRequest(WorkflowRequest):
    step: str                    # Investigation description
    step_number: int             # Current step (1-based)
    total_steps: int             # Estimated total steps
    next_step_required: bool     # Continue investigation?
    findings: str                # Discoveries this step
    files_checked: list[str]     # All examined files
    relevant_files: list[str]    # Files relevant to issue
    relevant_context: list[str]  # Methods/functions
    hypothesis: str              # Current theory
    confidence: str              # exploring/low/medium/high/very_high/almost_certain/certain
```

**Skill Implementation** (`pal_debug.py`):
```python
# Current: Simple single-call debug, no workflow
response = await provider.generate(prompt, ...)
```

**Missing Features:**
1. Multi-step workflow with `step_number`/`total_steps`/`next_step_required`
2. Confidence level tracking (7 levels)
3. Hypothesis evolution tracking
4. Expert analysis integration when confidence is high
5. "Certain" confidence skip logic (bypass external validation)

**Impact:** Cannot perform systematic root cause analysis

---

#### 1.3.3 Analyze Tool - **BROKEN**

**MCP Implementation** (`tools/analyze.py` - WorkflowTool subclass):
```python
class AnalyzeWorkflowRequest(WorkflowRequest):
    step: str
    step_number: int
    total_steps: int
    next_step_required: bool
    findings: str
    files_checked: list[str]
    relevant_files: list[str]
    relevant_context: list[str]
    issues_found: list[dict]     # Structured issues with severity
    analysis_type: Literal["architecture", "performance", "security", "quality", "general"]
    output_format: Literal["summary", "detailed", "actionable"]
```

**Skill Implementation** (`pal_analyze.py`):
```python
# Current: Simple single-call analysis
response = await provider.generate(prompt, ...)
```

**Missing Features:**
1. Multi-step analysis workflow
2. `analysis_type` parameter (5 types)
3. `output_format` parameter (3 formats)
4. Structured `issues_found` with severity levels
5. "Escalate to CodeReview" JSON response handling

**Impact:** Cannot perform comprehensive codebase analysis

---

#### 1.3.4 Conversation Memory - **INCOMPLETE**

**MCP Implementation** (`utils/conversation_memory.py`):
```python
class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: str
    files: Optional[list[str]]
    images: Optional[list[str]]
    tool_name: Optional[str]
    model_provider: Optional[str]
    model_name: Optional[str]
    model_metadata: Optional[dict]  # ❌ Missing in skill

class ThreadContext(BaseModel):
    thread_id: str
    parent_thread_id: Optional[str]  # ❌ Missing in skill
    created_at: str
    last_updated_at: str
    tool_name: str
    turns: list[ConversationTurn]
    initial_context: dict

# Features:
# - Newest-first file prioritization
# - Token-aware history building
# - Thread chaining via parent_thread_id
# - 3-hour TTL with auto-expiration
# - Model-specific token allocation
```

**Skill Implementation** (`lib/conversation.py`):
```python
class ConversationTurn:
    role: str
    content: str
    timestamp: str
    files: list[str]
    images: list[str]
    tool_name: str
    model_provider: str
    model_name: str
    # ❌ Missing: model_metadata

class ThreadContext:
    thread_id: str
    # ❌ Missing: parent_thread_id
    created_at: str
    last_updated_at: str
    tool_name: str
    turns: list[ConversationTurn]
    initial_context: dict
```

**Missing Features:**
1. `parent_thread_id` for thread chaining
2. `model_metadata` field
3. TTL/expiration mechanism
4. Newest-first file prioritization algorithm
5. Token-aware history building with model-specific limits
6. `get_thread_chain()` for traversing parent chains

---

#### 1.3.5 Prompt Discrepancies

| Prompt File | Issue | Severity |
|-------------|-------|----------|
| `consensus.md` | Missing `{stance_prompt}` placeholder | **Critical** |
| `consensus.md` | Missing 850-token limit mention | Medium |
| `debug.md` | Missing "TRACER TOOL INTEGRATION AWARENESS" | Medium |
| `debug.md` | Missing `no_bug_found` JSON schema | Medium |
| `analyze.md` | Missing "ESCALATE TO CODEREVIEW" JSON | Medium |
| `analyze.md` | Missing detailed "DELIVERABLE FORMAT" | Low |
| `codereview.md` | Missing "SCOPE TOO LARGE" JSON response | Medium |
| `thinkdeep.md` | Missing "KEY FOCUS AREAS" section | Low |

---

#### 1.3.6 Configuration Discrepancies

**config.yaml vs MCP config.py:**

| Setting | MCP Default | Skill Default | Issue |
|---------|-------------|---------------|-------|
| `temperature` | `1.0` (all tools) | `1.0` | ✅ Fixed |
| `model` | `"auto"` | `"auto"` | ✅ Fixed |
| `thinking_mode` | `"high"` (thinkdeep) | `"medium"` | ⚠️ Mismatch |
| `max_turns` | 50 | 50 | ✅ Match |
| `timeout_hours` | 3 | 3 | ✅ Match |

**Missing Configuration:**
- `MCP_PROMPT_SIZE_LIMIT` (60,000 chars)
- `DEFAULT_CONSENSUS_TIMEOUT` (120 seconds)
- Model-specific token allocation logic
- Provider priority order

---

#### 1.3.7 Clink Tool - **INCOMPLETE**

**MCP Implementation** (`tools/clink.py`):
```python
class CLinkRequest(BaseModel):
    prompt: str
    cli_name: str | None      # Configured CLI (gemini, claude, codex)
    role: str | None          # Role preset from config
    absolute_file_paths: list[str]
    images: list[str]
    continuation_id: str | None

# Features:
# - Dynamic CLI registry from conf/cli_clients
# - Role-based system prompt loading
# - Output truncation (MAX_RESPONSE_CHARS = 20,000)
# - <SUMMARY> tag extraction for large responses
# - Metadata pruning for MCP transport
```

**Skill Implementation** (`pal_clink.py`):
```python
# Current: Basic CLI bridging without role support
```

**Missing Features:**
1. Role configuration loading
2. Output size limiting (20,000 chars)
3. `<SUMMARY>` tag extraction
4. Dynamic CLI enumeration in schema

---

### 1.4 Provider Abstraction Gaps

**MCP Providers** (`providers/`):
| Provider | MCP | Skill | Gap |
|----------|-----|-------|-----|
| Gemini | Full | Basic | Medium |
| OpenAI | Full | Basic | Medium |
| OpenRouter | Full with aliases | Basic | Medium |
| X.AI (Grok) | Full | Partial | Medium |
| Azure OpenAI | Full | ❌ Missing | High |
| DIAL | Full | ❌ Missing | Medium |
| Custom (Ollama) | Full | ❌ Missing | High |

**Missing Provider Features:**
1. Model capability detection
2. Thinking mode support per-model
3. Token calculation per-model
4. Automatic model fallback

---

## Part 2: Implementation Plan

### Phase 1: Foundation Fixes (Priority: Critical)

#### 1.1 Fix Conversation Memory

**File:** `lib/conversation.py`

**Changes:**
```python
# Add missing fields to ConversationTurn
@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: str
    files: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    tool_name: str = ""
    model_provider: str = ""
    model_name: str = ""
    model_metadata: dict = field(default_factory=dict)  # ADD

# Add missing fields to ThreadContext
@dataclass
class ThreadContext:
    thread_id: str
    parent_thread_id: str = ""  # ADD for thread chaining
    created_at: str = ""
    last_updated_at: str = ""
    tool_name: str = ""
    turns: list = field(default_factory=list)
    initial_context: dict = field(default_factory=dict)

# Add thread chaining function
def get_thread_chain(thread_id: str, max_depth: int = 20) -> list[ThreadContext]:
    """Traverse parent chain to get all threads in sequence."""
    chain = []
    current_id = thread_id
    seen_ids = set()

    while current_id and len(chain) < max_depth:
        if current_id in seen_ids:  # Circular reference protection
            break
        seen_ids.add(current_id)

        context = get_thread(current_id)
        if not context:
            break
        chain.append(context)
        current_id = context.parent_thread_id

    chain.reverse()  # Chronological order
    return chain

# Add newest-first file prioritization
def get_conversation_file_list(context: ThreadContext) -> list[str]:
    """Extract unique files with newest-first prioritization."""
    if not context.turns:
        return []

    seen_files = set()
    file_list = []

    # Process turns in REVERSE order (newest first)
    for i in range(len(context.turns) - 1, -1, -1):
        turn = context.turns[i]
        if turn.files:
            for file_path in turn.files:
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    file_list.append(file_path)

    return file_list

# Add TTL expiration check
def is_thread_expired(context: ThreadContext, timeout_hours: int = 3) -> bool:
    """Check if thread has expired."""
    from datetime import datetime, timezone
    created = datetime.fromisoformat(context.created_at)
    now = datetime.now(timezone.utc)
    age_hours = (now - created).total_seconds() / 3600
    return age_hours > timeout_hours
```

---

#### 1.2 Fix Configuration

**File:** `config/config.yaml`

**Changes:**
```yaml
defaults:
  model: "auto"
  temperature: 1.0
  thinking_mode: "high"  # Change from "medium" to match MCP
  locale: ""

# Add missing limits
limits:
  prompt_size: 60000           # MCP_PROMPT_SIZE_LIMIT
  consensus_timeout: 120       # DEFAULT_CONSENSUS_TIMEOUT
  max_response_chars: 20000    # For clink output

# Add token allocation (model-specific)
token_allocation:
  default:
    context_window: 128000
    response_tokens: 8192
    file_percentage: 0.6
    history_percentage: 0.3
```

---

### Phase 2: Workflow Tools (Priority: Critical)

#### 2.1 Implement Workflow Base

**New File:** `lib/workflow.py`

```python
"""Workflow base class for multi-step investigation tools."""

from dataclasses import dataclass, field
from typing import Any, Optional
from abc import ABC, abstractmethod

@dataclass
class WorkflowRequest:
    """Base request for workflow tools."""
    prompt: str
    step: str
    step_number: int
    total_steps: int
    next_step_required: bool
    findings: str
    files_checked: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    relevant_context: list[str] = field(default_factory=list)
    model: str = "auto"
    continuation_id: str = ""
    absolute_file_paths: list[str] = field(default_factory=list)

@dataclass
class ConsolidatedFindings:
    """Accumulated findings across workflow steps."""
    findings: list[str] = field(default_factory=list)
    files_checked: set = field(default_factory=set)
    relevant_files: set = field(default_factory=set)
    relevant_context: list[str] = field(default_factory=list)
    issues_found: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    confidence: str = "low"

class WorkflowTool(ABC):
    """Base class for multi-step workflow tools."""

    def __init__(self):
        self.consolidated_findings = ConsolidatedFindings()

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_required_actions(self, step_number: int, confidence: str,
                             findings: str, total_steps: int) -> list[str]:
        pass

    @abstractmethod
    def should_call_expert_analysis(self, consolidated_findings) -> bool:
        pass

    def accumulate_step_data(self, request: WorkflowRequest):
        """Accumulate findings from current step."""
        self.consolidated_findings.findings.append(
            f"Step {request.step_number}: {request.findings}"
        )
        self.consolidated_findings.files_checked.update(request.files_checked)
        self.consolidated_findings.relevant_files.update(request.relevant_files)
        self.consolidated_findings.relevant_context.extend(request.relevant_context)

    def get_step_guidance(self, request: WorkflowRequest) -> str:
        """Generate guidance for next investigation step."""
        actions = self.get_required_actions(
            request.step_number,
            getattr(request, 'confidence', 'low'),
            request.findings,
            request.total_steps
        )

        if request.step_number == 1:
            return (
                f"MANDATORY: DO NOT call {self.get_name()} again immediately. "
                f"You MUST first investigate using appropriate tools. "
                f"Required actions:\n" +
                "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
            )
        elif request.next_step_required:
            return (
                f"PAUSE! Before calling {self.get_name()} step {request.step_number + 1}:\n" +
                "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
            )
        else:
            return "Investigation complete. Proceeding to expert analysis."
```

---

#### 2.2 Fix Debug Tool

**File:** `scripts/pal_debug.py`

**Add Workflow Parameters:**
```python
# Add to argument parser
parser.add_argument("--step", required=True, help="Investigation step description")
parser.add_argument("--step-number", type=int, required=True, help="Current step (1-based)")
parser.add_argument("--total-steps", type=int, required=True, help="Estimated total steps")
parser.add_argument("--next-step-required", type=bool, default=True)
parser.add_argument("--findings", required=True, help="Discoveries this step")
parser.add_argument("--files-checked", nargs="*", default=[])
parser.add_argument("--relevant-files", nargs="*", default=[])
parser.add_argument("--relevant-context", nargs="*", default=[])
parser.add_argument("--hypothesis", help="Current root cause theory")
parser.add_argument("--confidence",
    choices=["exploring", "low", "medium", "high", "very_high", "almost_certain", "certain"],
    default="low"
)

# Add workflow logic
async def execute_debug_workflow(args):
    workflow = DebugWorkflow()

    # Accumulate step data
    workflow.accumulate_step_data(args)

    # Check if we should skip expert analysis
    if args.confidence == "certain" and not args.next_step_required:
        return {
            "status": "certain_confidence_proceed_with_fix",
            "message": "Investigation complete with CERTAIN confidence. Proceed with fix.",
            "investigation": workflow.consolidated_findings
        }

    # Get step guidance
    if args.next_step_required:
        guidance = workflow.get_step_guidance(args)
        return {
            "status": "investigation_in_progress",
            "step": args.step_number,
            "guidance": guidance,
            "investigation_status": {
                "steps_completed": args.step_number,
                "files_examined": len(workflow.consolidated_findings.files_checked),
                "current_confidence": args.confidence
            }
        }

    # Call expert analysis
    if workflow.should_call_expert_analysis(workflow.consolidated_findings):
        expert_context = workflow.prepare_expert_context()
        expert_response = await call_external_model(expert_context)
        return {
            "status": "investigation_complete",
            "expert_analysis": expert_response,
            "investigation": workflow.consolidated_findings
        }
```

---

#### 2.3 Fix Analyze Tool

**File:** `scripts/pal_analyze.py`

**Add Missing Parameters:**
```python
# Add analysis_type and output_format
parser.add_argument("--analysis-type",
    choices=["architecture", "performance", "security", "quality", "general"],
    default="general"
)
parser.add_argument("--output-format",
    choices=["summary", "detailed", "actionable"],
    default="detailed"
)
parser.add_argument("--issues-found", type=json.loads, default=[],
    help="Structured issues with severity levels"
)

# Add workflow step parameters (same as debug)
parser.add_argument("--step", required=True)
parser.add_argument("--step-number", type=int, required=True)
parser.add_argument("--total-steps", type=int, required=True)
parser.add_argument("--next-step-required", type=bool, default=True)
parser.add_argument("--findings", required=True)
```

---

#### 2.4 Fix Consensus Tool

**File:** `scripts/pal_consensus.py`

**Complete Rewrite Required:**
```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27", "pyyaml>=6.0"]
# ///

"""PAL Consensus - Multi-model perspective gathering with stance steering."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from config import load_config
from providers import get_provider
from conversation import get_thread, add_turn, create_thread

# Stance prompts (from MCP consensus_prompt.py)
STANCE_PROMPTS = {
    "for": """
PERSPECTIVE: ADVOCATE
You are taking a supportive stance toward this proposal. Your role is to:
- Highlight the strengths and potential benefits
- Identify opportunities and positive outcomes
- Find solutions to challenges raised
- Emphasize feasibility and value
While maintaining intellectual honesty, lean toward seeing how this could work.
""",
    "against": """
PERSPECTIVE: CRITIC
You are taking a skeptical stance toward this proposal. Your role is to:
- Identify weaknesses, risks, and potential pitfalls
- Challenge assumptions and question feasibility
- Highlight costs, complexity, and maintenance burden
- Consider what could go wrong
While being constructive, lean toward finding legitimate concerns.
""",
    "neutral": """
PERSPECTIVE: BALANCED ANALYST
You are providing an objective, balanced assessment. Your role is to:
- Weigh both benefits and risks equally
- Present multiple viewpoints fairly
- Avoid leaning toward approval or rejection
- Focus on factual analysis over advocacy
Provide a balanced view that helps inform the decision.
"""
}

async def query_model_with_stance(provider, prompt: str, stance: str,
                                   system_prompt: str, files: list[str]) -> dict:
    """Query a single model with stance-adjusted prompt."""
    # Inject stance into system prompt
    stance_prompt = STANCE_PROMPTS.get(stance, STANCE_PROMPTS["neutral"])
    full_system_prompt = system_prompt.replace("{stance_prompt}", stance_prompt)

    response = await provider.generate(
        prompt=prompt,
        system_prompt=full_system_prompt,
        files=files,
        max_tokens=850  # Consensus responses are limited
    )

    return {
        "stance": stance,
        "model": provider.model_name,
        "response": response
    }

async def gather_consensus(args):
    """Query multiple models and synthesize responses."""
    config = load_config()

    # Determine which models to query
    models_to_query = args.models or ["gemini-2.5-flash", "gpt-4o-mini"]
    stances = ["for", "against", "neutral"] if args.full_spectrum else [args.stance]

    # Load system prompt
    prompt_path = Path(__file__).parent.parent / "prompts" / "consensus.md"
    system_prompt = prompt_path.read_text() if prompt_path.exists() else ""

    # Query all model/stance combinations
    tasks = []
    for model in models_to_query:
        for stance in stances:
            provider = get_provider(model)
            tasks.append(query_model_with_stance(
                provider, args.prompt, stance, system_prompt, args.files
            ))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful responses
    valid_responses = [r for r in responses if not isinstance(r, Exception)]

    # Synthesize consensus
    synthesis = synthesize_responses(valid_responses, args.prompt)

    return {
        "status": "success",
        "individual_responses": valid_responses,
        "synthesis": synthesis,
        "models_queried": models_to_query,
        "stances_used": stances
    }

def synthesize_responses(responses: list[dict], original_prompt: str) -> dict:
    """Synthesize multiple model responses into consensus."""
    # Group by stance
    by_stance = {"for": [], "against": [], "neutral": []}
    for r in responses:
        by_stance[r["stance"]].append(r)

    # Extract common themes
    all_verdicts = [extract_verdict(r["response"]) for r in responses]
    all_confidence = [extract_confidence(r["response"]) for r in responses]

    avg_confidence = sum(c for c in all_confidence if c) / len(all_confidence) if all_confidence else 5

    return {
        "overall_verdict": determine_overall_verdict(by_stance),
        "average_confidence": round(avg_confidence, 1),
        "key_agreements": find_agreements(responses),
        "key_disagreements": find_disagreements(responses),
        "recommendation": generate_recommendation(by_stance, avg_confidence)
    }

# ... (implement helper functions)
```

---

### Phase 3: Prompt Synchronization (Priority: Medium)

#### 3.1 Update Consensus Prompt

**File:** `prompts/consensus.md`

**Add Missing Content:**
```markdown
# At the top, add stance placeholder
PERSPECTIVE FRAMEWORK
{stance_prompt}

# At the end, add token limit reminder
REMINDERS
...
- Keep your response concise - your entire reply must not exceed 850 tokens
- CRITICAL: Your stance does NOT override your responsibility to provide truthful guidance
```

---

#### 3.2 Update Debug Prompt

**File:** `prompts/debug.md`

**Add Missing Sections:**
```markdown
# Add after CRITICAL LINE NUMBER INSTRUCTIONS
TRACER TOOL INTEGRATION AWARENESS
If the agent used the tracer tool during investigation, the findings will include:
- Method call flow analysis
- Class dependency mapping
- Side effect identification
- Execution path tracing

# Add no_bug_found JSON schema
IF NO BUG FOUND AFTER THOROUGH INVESTIGATION:
{
  "status": "no_bug_found",
  "summary": "<summary of investigation>",
  "investigation_steps": ["<step 1>", "<step 2>"],
  "areas_examined": ["<code areas>"],
  "confidence_level": "High|Medium|Low",
  "alternative_explanations": ["<possible misunderstanding>"],
  "recommended_questions": ["<clarifying questions>"],
  "next_steps": ["<suggested actions>"]
}
```

---

#### 3.3 Update Analyze Prompt

**File:** `prompts/analyze.md`

**Add Missing Content:**
```markdown
# Add escalation JSON
ESCALATE TO A FULL CODEREVIEW IF REQUIRED
If a comprehensive code-base–wide review is essential, respond ONLY with:
{
  "status": "full_codereview_required",
  "important": "Please use pal's codereview tool instead",
  "reason": "<brief rationale for escalation>"
}
```

---

### Phase 4: Provider Enhancements (Priority: Medium)

#### 4.1 Add Model Capabilities

**File:** `lib/providers.py`

```python
# Add model capability detection
MODEL_CAPABILITIES = {
    "gemini-2.5-flash": {
        "context_window": 1048576,
        "thinking_modes": ["minimal", "low", "medium", "high", "max"],
        "supports_images": True,
        "supports_json_mode": True
    },
    "gemini-2.5-pro": {
        "context_window": 2097152,
        "thinking_modes": ["minimal", "low", "medium", "high", "max"],
        "supports_images": True,
        "supports_json_mode": True
    },
    "gpt-4o": {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True
    },
    "gpt-4o-mini": {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True
    },
    "o3-mini": {
        "context_window": 200000,
        "thinking_modes": ["low", "medium", "high"],
        "supports_images": False,
        "supports_json_mode": True
    },
    "claude-sonnet-4-20250514": {
        "context_window": 200000,
        "thinking_modes": [],
        "supports_images": True,
        "supports_json_mode": True
    }
}

def get_model_capabilities(model_name: str) -> dict:
    """Get capabilities for a model."""
    # Check exact match
    if model_name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[model_name]

    # Check prefix match for versioned models
    for name, caps in MODEL_CAPABILITIES.items():
        if model_name.startswith(name.split("-")[0]):
            return caps

    # Default capabilities
    return {
        "context_window": 128000,
        "thinking_modes": [],
        "supports_images": False,
        "supports_json_mode": True
    }
```

---

#### 4.2 Add Custom Provider (Ollama)

**File:** `lib/providers.py`

```python
class CustomProvider(BaseProvider):
    """Provider for Ollama and other OpenAI-compatible APIs."""

    def __init__(self, base_url: str, api_key: str = "", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 1.0),
                    "max_tokens": kwargs.get("max_tokens", 8192)
                },
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
```

---

### Phase 5: Missing Tools (Priority: Low-Medium)

#### 5.1 Add Planner Tool

**New File:** `scripts/pal_planner.py`

Implement using workflow pattern similar to debug/analyze.

#### 5.2 Add Other Missing Tools

Create stubs for remaining tools:
- `pal_precommit.py`
- `pal_secaudit.py`
- `pal_docgen.py`
- `pal_refactor.py`
- `pal_testgen.py`
- `pal_tracer.py`
- `pal_challenge.py`
- `pal_apilookup.py`
- `pal_listmodels.py`
- `pal_version.py`

---

## Part 3: Implementation Priority

### Immediate (Week 1)
1. ✅ Fix conversation memory (add `parent_thread_id`, `model_metadata`)
2. ✅ Fix config.yaml (`thinking_mode: "high"`)
3. 🔴 Fix consensus tool (critical - completely broken)

### Short-term (Week 2)
4. 🔴 Implement workflow base class
5. 🔴 Fix debug tool (add workflow parameters)
6. 🔴 Fix analyze tool (add workflow parameters)
7. 🟡 Sync all prompts with MCP versions

### Medium-term (Week 3-4)
8. 🟡 Add model capabilities detection
9. 🟡 Add Custom/Ollama provider
10. 🟡 Fix clink tool (role support, output limiting)
11. 🟢 Add planner tool
12. 🟢 Add remaining tools

### Legend
- 🔴 Critical - Feature broken or missing
- 🟡 Medium - Feature incomplete
- 🟢 Low - Nice to have

---

## Part 4: Testing Strategy

### Unit Tests
```bash
# Test conversation memory
python -m pytest tests/test_conversation.py -v

# Test workflow base
python -m pytest tests/test_workflow.py -v

# Test consensus multi-model
python -m pytest tests/test_consensus.py -v
```

### Integration Tests
```bash
# Run with real API calls
./run_integration_tests.sh

# Quick validation
python communication_simulator_test.py --quick
```

### Specific Test Cases
1. **Consensus stance steering**: Verify `for`/`against`/`neutral` produce different responses
2. **Debug workflow**: Verify multi-step investigation with confidence progression
3. **Thread chaining**: Verify `parent_thread_id` traversal works
4. **File prioritization**: Verify newest files take precedence

---

## Part 5: Success Criteria

### Feature Parity Checklist
- [ ] Consensus tool queries multiple models with stance steering
- [ ] Debug tool supports multi-step workflow with confidence levels
- [ ] Analyze tool supports analysis_type and output_format
- [ ] Conversation memory supports thread chaining
- [ ] Conversation memory implements newest-first file prioritization
- [ ] All prompts match MCP system prompts
- [ ] Config defaults match MCP defaults
- [ ] Clink supports role configuration and output limiting

### Performance Targets
- Consensus multi-model query: < 60 seconds
- Conversation history build: < 1 second
- File embedding: < 5 seconds for 10 files

---

## Appendix A: File Reference

### Files to Modify
| File | Changes |
|------|---------|
| `lib/conversation.py` | Add thread chaining, file prioritization |
| `lib/providers.py` | Add model capabilities, custom provider |
| `config/config.yaml` | Fix thinking_mode, add limits |
| `scripts/pal_consensus.py` | Complete rewrite for multi-model |
| `scripts/pal_debug.py` | Add workflow parameters |
| `scripts/pal_analyze.py` | Add workflow parameters |
| `prompts/consensus.md` | Add stance_prompt, token limit |
| `prompts/debug.md` | Add tracer awareness, no_bug_found |
| `prompts/analyze.md` | Add escalation JSON |

### New Files to Create
| File | Purpose |
|------|---------|
| `lib/workflow.py` | Workflow base class |
| `scripts/pal_planner.py` | Planner tool |
| `tests/test_workflow.py` | Workflow tests |
| `tests/test_consensus.py` | Consensus tests |

---

## Appendix B: MCP Reference Paths

Key MCP files for reference:
- `tools/consensus.py` - Multi-model consensus implementation
- `tools/debug.py` - Debug workflow implementation
- `tools/analyze.py` - Analyze workflow implementation
- `tools/workflow/base.py` - WorkflowTool base class
- `utils/conversation_memory.py` - Conversation memory with all features
- `systemprompts/*.py` - All system prompts
- `config.py` - Configuration constants
- `server.py` - Tool registry and MCP handlers
