# Multi-Turn Conversation Patterns

Using `continuation_id` for stateful multi-turn interactions.

## Basic Continuation

```bash
# First call - starts new thread
result=$(uv run scripts/pal_chat.py --prompt "Explain this code" --files main.py --json)
CID=$(echo $result | jq -r '.continuation_id')
# CID: "abc-123-..."

# Second call - continues thread
uv run scripts/pal_chat.py --prompt "What about error handling?" \
  --continuation-id $CID --json
# Full context from first call available
```

## Cross-Tool Continuation

The same `continuation_id` works across different tools:

```bash
# Start with chat
result=$(uv run scripts/pal_chat.py --prompt "Review this code" --files auth.py --json)
CID=$(echo $result | jq -r '.continuation_id')

# Escalate to codereview - same context
uv run scripts/pal_codereview.py --files auth.py --continuation-id $CID --json

# Then to debug - still same context
uv run scripts/pal_debug.py --issue "Investigate SQL issue" \
  --files auth.py --continuation-id $CID --json
```

## Context Accumulation

Each turn adds to the conversation:
- User prompts
- AI responses
- Files examined
- Tool attributions
- Model/provider info

## Token-Aware History

History is built within model's context window:
- Recent turns prioritized
- Old turns dropped if needed
- File references deduplicated (newest first)

## Continuation Best Practices

1. **Use continuation for iterative work**
   ```bash
   # Iterative debugging
   pal_debug --issue "..." --files src/ → CID
   # Read some files...
   pal_debug --issue "Check hypothesis" --continuation-id CID
   ```

2. **Start fresh for unrelated tasks**
   ```bash
   # New task = no continuation_id
   pal_chat --prompt "Different topic"
   ```

3. **Cross-tool for escalation**
   ```bash
   # Same CID when escalating
   pal_codereview → pal_debug → pal_thinkdeep
   ```

## Storage Options

Configure in `config/config.yaml`:

```yaml
conversation:
  max_turns: 50
  timeout_hours: 3
  storage: "memory"  # or "sqlite" for persistence
```

- `memory`: Fast, resets on restart
- `sqlite`: Persists across sessions
