# PAL Skills - Example Workflows

This document shows common workflows using PAL skills.

## 1. Multi-Turn Development Discussion

```bash
# Start a discussion about a feature
CONT_ID=$(python scripts/pal_chat.py \
  --prompt "I'm thinking about adding caching to our API. What approach would you recommend?" \
  --files src/api/routes.py \
  --json | jq -r '.continuation_id')

# Continue with more context
python scripts/pal_chat.py \
  --prompt "What about Redis vs in-memory caching? We have about 10K requests/second." \
  --continuation-id $CONT_ID

# Dive deeper
python scripts/pal_chat.py \
  --prompt "How would we handle cache invalidation for user-specific data?" \
  --continuation-id $CONT_ID
```

## 2. Deep Architectural Analysis

```bash
# Analyze a complex system
python scripts/pal_thinkdeep.py \
  --prompt "Analyze the trade-offs of our current microservices architecture.
            We're experiencing latency issues and considering consolidation." \
  --files src/services/ \
  --thinking-mode max
```

## 3. Pre-Commit Code Review

```bash
# Get changed files
CHANGED_FILES=$(git diff --name-only HEAD~1 -- "*.py")

# Review all changes
python scripts/pal_codereview.py \
  --files $CHANGED_FILES \
  --focus security performance \
  --json > review_results.json
```

## 4. Cross-Tool Continuation

```bash
# Start with chat
CONT_ID=$(python scripts/pal_chat.py \
  --prompt "Let's review the authentication system" \
  --files src/auth/ \
  --json | jq -r '.continuation_id')

# Switch to code review (same conversation)
python scripts/pal_codereview.py \
  --files src/auth/oauth.py \
  --focus security \
  --continuation-id $CONT_ID

# Deep dive with thinkdeep (same conversation)
python scripts/pal_thinkdeep.py \
  --prompt "Based on the review, what's the best way to fix the token refresh issue?" \
  --continuation-id $CONT_ID
```

## 5. Using Multiple AI CLIs

```bash
# Get Gemini's perspective (with web search)
python scripts/pal_clink.py \
  --cli gemini \
  --prompt "What's the latest best practice for JWT token rotation in 2024?" \
  --json > gemini_response.json

# Get Claude's perspective
python scripts/pal_clink.py \
  --cli claude \
  --role codereviewer \
  --prompt "Review this JWT implementation" \
  --files src/auth/jwt.py \
  --json > claude_response.json
```

## 6. Iterative Debugging

```bash
# Start debugging session
CONT_ID=$(python scripts/pal_chat.py \
  --prompt "We're seeing intermittent 500 errors in production. Here's the stack trace: [trace]" \
  --files src/api/handlers.py src/db/connection.py \
  --json | jq -r '.continuation_id')

# Add more context as you investigate
python scripts/pal_chat.py \
  --prompt "I found that the errors correlate with high memory usage. Here's the memory profile:" \
  --files logs/memory_profile.txt \
  --continuation-id $CONT_ID

# Get final recommendations
python scripts/pal_thinkdeep.py \
  --prompt "Given everything we've discussed, what's the root cause and fix?" \
  --continuation-id $CONT_ID \
  --thinking-mode high
```

## 7. Planning a Feature

```bash
# Use Gemini CLI as planner
python scripts/pal_clink.py \
  --cli gemini \
  --role planner \
  --prompt "Plan the implementation of a rate limiting system for our API.
            Requirements:
            - Per-user and per-IP limits
            - Configurable time windows
            - Redis-backed storage
            - Graceful degradation" \
  --files src/api/middleware.py
```

## Tips

1. **Save continuation IDs** for long-running discussions
2. **Use `--json`** for scripting and automation
3. **Combine tools** - chat for exploration, thinkdeep for decisions, codereview for validation
4. **Use focus areas** in codereview to get targeted feedback
5. **Set thinking-mode high** for complex architectural questions
