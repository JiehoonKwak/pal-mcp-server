# Security Audit Workflow

Comprehensive security review using PAL tools.

## Workflow Steps

```
secaudit → codereview (security focus) → fix → testgen → precommit
```

## Example

### 1. Initial Security Audit
```bash
result=$(uv run scripts/pal_secaudit.py \
  --files src/api/ src/auth/ \
  --json)

# Check findings
echo $result | jq '.agentic.confidence'
# Output: "high"

CID=$(echo $result | jq -r '.continuation_id')
```

### 2. Deep Dive on Critical Issues
```bash
# If critical issues found, get detailed code review
uv run scripts/pal_codereview.py \
  --files src/auth/jwt.py src/api/user_routes.py \
  --focus security \
  --continuation-id $CID \
  --json
```

### 3. Investigate Specific Vulnerability
```bash
uv run scripts/pal_debug.py \
  --issue "SQL injection vulnerability in user search" \
  --files src/api/user_routes.py src/db/queries.py \
  --continuation-id $CID
```

### 4. Get Second Opinion
```bash
uv run scripts/pal_consensus.py \
  --proposal "Proposed fix: Use parameterized queries for all user input" \
  --models gemini-2.5-pro gpt-4o \
  --stances for against \
  --files src/db/queries.py \
  --continuation-id $CID
```

### 5. Implement Fixes
Apply the security fixes based on recommendations.

### 6. Generate Security Tests
```bash
uv run scripts/pal_testgen.py \
  --files src/api/user_routes.py src/db/queries.py \
  --prompt "Generate security tests: SQL injection, auth bypass, XSS" \
  --continuation-id $CID
```

### 7. Final Validation
```bash
uv run scripts/pal_precommit.py \
  --files src/api/ src/auth/ src/db/ tests/ \
  --continuation-id $CID
```

## OWASP Top 10 Coverage

The secaudit tool covers:
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A04: Insecure Design
- A05: Security Misconfiguration
- A06: Vulnerable Components
- A07: Authentication Failures
- A08: Data Integrity Failures
- A09: Logging Failures
- A10: SSRF
