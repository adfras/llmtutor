# Software Technical Standard (STS) — Real‑Time Psych Tutor

Version: 1.0
Status: Active (living document)
Owners: Engineering (Server/LLM), Product, Security

## 1) Scope & Purpose

This STS defines the technical standards, interfaces, data contracts, security, quality bars, and operational practices for the Real‑Time LLM‑Powered Psychology Tutor. It ensures consistent implementation, safe extensibility, and predictable behavior across contributors and deployments.

## 2) References

- Runbook: `AGENTS.md`
- Architecture: `docs/rt-psych-tutor/architecture.md`
- API Spec (high‑level): `docs/rt-psych-tutor/api_spec.yaml`
- Skill Map: `docs/rt-psych-tutor/skill_map.psych101.yaml`
- Privacy/Security: `docs/rt-psych-tutor/privacy_security.md`
- Source of truth for endpoints: `server/app.py`

## 3) Definitions

- Skill: Leaf or internal node in the PSY101 skill map.
- Category: The top‑level ancestor of a skill (used for roll‑ups).
- MCQ Minimal: Question with `{stem, options[], correct_index}` only.
- MCQ Rich: Includes `rationales[]` and `misconception_tags[]` aligned to options.
- Mastery: Per‑skill continuous score in [0,1], updated after each attempt with decay.

## 4) Architecture & Components

- Web UI: Static `web/index.html`; no build step; talks to REST API.
- API Server: FastAPI (`server/app.py`); serves static UI + REST endpoints.
- LLM Wrapper: `tutor/llm_openai.py`; fixed model; strict JSON responses.
- Storage: JSON file `data/user_stats.json` via `server/storage.py` (thread‑safe).
- Skill Map: YAML loaded via `tutor/skill_map.py`.

## 5) Model & Prompting Standards

- Model: Hard‑locked to `gpt-5-nano-2025-08-07`. Do not override by env, flags, or API.
- Response format: Always request `response_format={"type":"json_object"}` and include the word “JSON” in prompts to reinforce compliance.
- MCQ Minimal Prompt: Must produce only `{stem, options[=num_options], correct_index}`.
- MCQ Rich Prompt: Must additionally produce `{question_id, rationales[], misconception_tags[], difficulty}` with `rationales.length == options.length == misconception_tags.length`.
- SAQ/Grading: JSON‑only outputs; no prose outside JSON.

## 6) HTTP API (Authoritative Contracts)

Base path: `/api` (no version prefix in MVP; add `/v1` when breaking changes are introduced).

1. GET `/api/skills`
   - Response: `{ skills: [{ id, name, bloom }] }`

2. POST `/api/generate`
   - Request: `{ skill_id, type:"mcq"|"saq", difficulty:"easy"|"medium"|"hard", num_options:int, rich?:boolean, verify?:boolean, use_templates?:boolean, user_id?:string }`
   - Response (mcq minimal): `{ type:"mcq", skill_id, question:{ stem, options[], correct_index } }`
   - Response (mcq rich): `{ type:"mcq", skill_id, question:{ question_id, stem, options[], correct_index, rationales[], misconception_tags[], difficulty } }`
   - Notes: Minimal path uses per‑key prefetch cache; rich path bypasses cache.

3. POST `/api/record`
   - Request: `{ user_id:string, skill_id:string, correct:boolean, item_id?:string, confidence?:1..5, time_to_answer_ms?:int, misconception_tag?:string }`
   - Response: `{ ok:true, stats, category_id }`
   - Behavior: Updates per‑skill counts, mastery (EMA + decay), `last_seen_at`, and optional misconception counter; also updates per‑category counts.
   - Side effects: If `item_id` provided, aggregates item‑level stats (delivered, correct/wrong, confidence/time sums).

4. GET `/api/user/{user_id}/stats`
   - Response: `{ per_skill: {skill_id: {correct, wrong, total, mastery?, last_seen_at?, misconceptions?}}, per_category: {category_id: {correct, wrong, total}} }`

5. POST `/api/user/upsert`
   - Request: `{ username:string }`
   - Response: `{ user_id, username }` (case‑insensitive idempotence).

6. GET `/api/user/by-name/{username}`
   - Response: `{ user_id, username, stats }` or 404 if unknown.

7. POST `/api/next` (Adaptive)
   - Request: `{ user_id:string, current_skill_id?:string, last_correct?:boolean, type:"mcq", difficulty, num_options, rich?:boolean, verify?:boolean, use_templates?:boolean }`
   - Response: `{ type:"mcq", skill_id, reason:"remediation"|"continue-current"|"review-due"|"advance-new"|"fallback", question }`

Errors: Always use structured errors via FastAPI HTTPException: `{ detail: string }`.

8. POST `/api/item/flag`
   - Request: `{ item_id:string }`
   - Response: `{ ok:true, item:{ item_id, flags:int } }`
   - Behavior: Increments a per‑item flag counter for QA.

## 7) Data Standards

Skill Map YAML:

- Required keys per skill: `id`, `name`; optional: `parent`, `prereqs[]`, `bloom`, `description`.
- Category Calculation: top‑level ancestor discovered by following `parent` links until root.

User Stats JSON (persisted):

```
{
  "users": { "<user_id>": { "username": "Alice" }, ... },
  "usernames": { "alice": "<user_id>", ... },
  "stats": {
    "<user_id>": {
      "per_skill": {
        "<skill_id>": {
          "correct": int, "wrong": int, "total": int,
          "mastery": float [0..1],
          "last_seen_at": ISO8601 or null,
          "misconceptions": { "<tag>": count, ... }
        }
      },
      "per_category": { "<category_id>": { "correct": int, "wrong": int, "total": int } }
    }
  }
  ,
  "items": {
    "<item_id>": {
      "skill_id": "...",
      "created_at": ISO,
      "last_seen_at": ISO,
      "delivered": int,
      "correct": int,
      "wrong": int,
      "confidence_sum": int,
      "confidence_n": int,
      "time_sum_ms": int,
      "time_n": int,
      "flags": int,
      "prompt_id"?: string,
      "seed"?: int
    }
  }
}
```

Mastery Update Rule:

- On record: `decayed = mastery * exp(-ln(2) * days / half_life_days)` with `half_life_days = 7`.
- EMA update: `new = alpha*decayed + (1-alpha)*result`, with `alpha = 0.7`, `result = 1.0 if correct else 0.0`.
- Clamp to [0,1]; set `last_seen_at = now`.

Adaptive Policy (when selecting next):

1) If last answer wrong and `current_skill_id` present → remediation on current skill (one immediate follow‑up allowed).
2) Else, continue current only when the skill is due (time ≥ interval) and decayed mastery < 0.8.
3) Else, find review‑due skills using mastery‑based intervals (1/3/7/14 days bands) and select the lowest‑mastery, longest‑waiting.
4) Else, advance to next new skill with prerequisites satisfied (prefer siblings).

## 8) Security & Privacy

- API Key: `OPENAI_API_KEY` via environment or `.env`; never log or expose the key.
- PII: Only store `username`; prohibit storing free‑text answers by default; SAQ inputs should not persist unless explicitly enabled and documented.
- Logs: Avoid logging request bodies that may contain student inputs; keep minimal metadata.
- Transport: Use HTTPS in production. Configure CORS narrowly if serving UI and API from different origins.
- Secrets: `.env` excluded from VCS; use environment variables in production.

## 9) Non‑Functional Requirements (Targets)

- Latency: MCQ minimal P95 ≤ 2.5s (end‑to‑end) with prefetch on warm cache; P99 ≤ 4s.
- Availability: 99.5% monthly for API endpoints (best‑effort in MVP).
- Throughput: Support at least 10 RPS on `/api/generate` with graceful degradation (backoff if LLM throttles).
- Error Budget: 1% 5xx over rolling 24h.

## 10) Quality Gates

- Linting/Style: Python 3.10+; Black + Ruff recommended; keep functions small and clear.
- JSON Contracts: Maintain backward compatibility; additive changes only for MVP (use `rich` flag for larger payloads).
- Unit Tests (recommended):
  - `server/storage.py`: record(), mastery decay, misconception counters.
  - `server/app.py`: `_top_category_id`, `/api/record` basic path.
  - JSON schema checks for MCQ minimal.
- Integration Tests (smoke): `/api/skills`, `/api/generate`, `/api/record`, `/api/next` with a mocked LLM client.
 - Validation: MCQ rule‑based validator must pass or retry; server must attach `item_id`.

## 11) Observability

- Metrics (recommended): request counts, latencies, LLM call counts/errors, cache hit ratio for prefetch.
- Tracing (optional): annotate LLM round‑trips and background prefetch tasks.
- Health Checks: readiness = loads skill map; liveness = process alive.

## 12) Deployment & Operations

- Dev: `uvicorn server.app:app --reload`.
- Prod: Run behind a reverse proxy (e.g., Nginx) with HTTPS termination; configure timeouts ≥ LLM P99.
- Prefetch: Enabled for minimal MCQs; cap per‑key queue length to 2.
- Backpressure: If LLM errors, return 503 with retry‑after where appropriate; UI should show a retry prompt.

## 13) Backward Compatibility & Versioning

- API Changes: Use additive fields; breaking changes require a versioned prefix (e.g., `/api/v2`).
- Storage: New per‑skill fields (`mastery`, `last_seen_at`, `misconceptions`) must be optional with sensible defaults when absent.

## 14) Security Review Checklist

- [ ] `.env` not committed; secrets stored securely.
- [ ] No PII beyond username persisted.
- [ ] Input validation on all endpoints; unknown `skill_id` → 404; required fields enforced.
- [ ] CORS restricted in production.
- [ ] Logs free of keys and student free‑text.

## 15) Change Management (Definition of Done)

- Code aligns with this STS and repo constraints (fixed model, JSON‑only).
- Contracts: API and storage changes documented here and in README.
- Tests updated/passing locally (where present); manual smoke checks documented.
- Operational notes added for new endpoints/features.

---

Annex A — Normative JSON Schemas

1) MCQ Minimal `question`

```
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.org/schemas/mcq-minimal.json",
  "type": "object",
  "required": ["stem", "options", "correct_index"],
  "properties": {
    "stem": {"type": "string", "minLength": 8, "maxLength": 200},
    "options": {
      "type": "array",
      "minItems": 2,
      "maxItems": 7,
      "items": {"type": "string", "minLength": 1, "maxLength": 64}
    },
    "correct_index": {"type": "integer", "minimum": 0},
    "item_id": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    "prompt_id": {"type": "string"}
  },
  "additionalProperties": true
}
```

2) MCQ Rich additions

```
{
  "rationales": {"type": "array", "items": {"type": "string"}},
  "misconception_tags": {"type": "array", "items": {"type": "string"}},
  "template_id": {"type": "string"}
}
```

3) User Stats per‑skill entry

```
{
  "correct": {"type":"integer","minimum":0},
  "wrong": {"type":"integer","minimum":0},
  "total": {"type":"integer","minimum":0},
  "mastery": {"type":"number","minimum":0,"maximum":1},
  "last_seen_at": {"type":"string"},
  "misconceptions": {"type":"object","additionalProperties": {"type":"integer","minimum":0}},
  "template_counts": {"type":"object","additionalProperties": {"type":"integer","minimum":0}}
}
```

4) Items aggregation entry

```
{
  "skill_id": {"type":"string"},
  "delivered": {"type":"integer","minimum":0},
  "correct": {"type":"integer","minimum":0},
  "wrong": {"type":"integer","minimum":0},
  "confidence_sum": {"type":"integer","minimum":0},
  "confidence_n": {"type":"integer","minimum":0},
  "time_sum_ms": {"type":"integer","minimum":0},
  "time_n": {"type":"integer","minimum":0},
  "flags": {"type":"integer","minimum":0},
  "p_value": {"type":"number","minimum":0,"maximum":1},
  "flag_rate": {"type":"number","minimum":0,"maximum":1},
  "retired": {"type":"boolean"}
}
```

Annex B — Algorithms (Reference)

- Mastery decay: `m_decayed = m_prev * exp(-ln(2) * dt_days / 7)`.
- EMA update: `m_new = 0.7 * m_decayed + 0.3 * y`, `y ∈ {0,1}`.
- Review interval bands (days): `<0.4 → 1`, `<0.6 → 3`, `<0.8 → 7`, `≥0.8 → 14`.
- Next skill selection priority: (1) remediation; (2) continue current if due and <0.8; (3) due review with lowest mastery (tie longest wait); (4) advance to new eligible (prereqs ≥0.8; prefer siblings).
- Item retirement rule: after `delivered ≥ 100`, retire if `flag_rate > 0.01` or `p_value < 0.1` or `p_value > 0.9`.

Annex C — Feature Flags & Config

- `TUTOR_VERIFY` (env): default self‑verification on rich path when set to `1`.
- `TUTOR_USE_TEMPLATES` (env): prefer curated templates when set to `1`.
- Request flags override env per call.

Annex D — Logging & Metrics (Recommended)

- Logs: method, path, status, latency_ms, user_id (hash), skill_id, item_id, cache_hit, llm_latency_ms, validator_result.
- Metrics: `http_requests_total{path,status}`, `http_request_duration_seconds_bucket`, `llm_calls_total{type}`, `validator_failures_total`, `item_flag_rate`, `cache_hits_total`.

Annex E — Error Codes

- 400: missing required fields; invalid type; invalid `skill_id` in body.
- 404: unknown resource (`/api/user/by-name/{username}`, `/api/skill/{skill_id}/oer`).
- 503: transient upstream (LLM) failure (recommended for production hardening).

Annex F — Test Strategy

- Unit: mastery update with synthetic timestamps; validator edge cases; item_id determinism.
- Integration (mock LLM): check `/api/generate` minimal/rich shapes; `/api/record` state deltas; `/api/next` policy transitions.
- Accessibility: tab/enter flows; `aria-live` announcements present.
