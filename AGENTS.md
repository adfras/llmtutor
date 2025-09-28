# Agent Runbook — Real‑Time Psych Tutor

This document guides an automation agent or contributor on how to operate and extend the project.

## Purpose

Deliver a minimal, fast, LLM‑powered psychology tutor that:
- Generates MCQs on demand.
- Records per‑user correctness by skill and category across sessions.
- Uses a fixed OpenAI model (`gpt-5-nano-2025-08-07`) with strict JSON outputs.
- Supports a provable novice mode (anonymization + evidence‑gated scoring) so a small OS LLM “student” must learn from session NOTES, not pretraining.

## Operating Instructions

1) Environment
- Ensure `.env` contains `OPENAI_API_KEY`.
- Python venv recommended: `python3 -m venv .venv && source .venv/bin/activate`.

2) Install & Run
- `pip install -r requirements.txt`
- `uvicorn server.app:app --reload`
- Open http://localhost:8000
 - UI controls:
   - Show explanations (default on): brief rationales after answering.
   - Self‑verify: second model pass answers the item; regenerates on mismatch (slower).
   - Use templates: curated, misconception‑aligned patterns per skill; rotates least‑used first.
   - Mode: Practice (immediate feedback) vs Check‑up (10‑item block, delayed summary).

3) Health Checks
- `GET /api/skills` should return JSON with a list of skills.
- `POST /api/generate` with `{ "skill_id":"cog-learning-theories", "type":"mcq", "num_options":5 }` returns `{ question: { stem, options, correct_index } }`.
 - `POST /api/next` with `{ "user_id":"...", "current_skill_id":"cog-learning-theories" }` returns `{ skill_id, reason, question }`.
 - `GET /api/skill/{skill_id}/oer` returns a learn‑more link; `POST /api/item/flag` increments an item flag.
 - `POST /api/next` with `{ "user_id":"...", "current_skill_id":"cog-learning-theories" }` returns an adaptive next MCQ with `{ skill_id, reason, question }`.

4) Key Verification (CLI)
- `python -m tutor.cli verify` confirms model and key access.

Synthetic Student (DeepInfra/OpenAI)
- Ensure `.env` contains `DEEPINFRA_API_KEY`.
- Run the harness to simulate users answering questions:
  - `export $(grep -v '^#' .env | xargs)`
  - `python student_bot_deepinfra.py --username bot01 --n 300 --provider deepinfra --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`
  - Closed‑book novice: add `--closed-book --notes-file data/notes_<user>.txt` and start the server with `TUTOR_ANONYMIZE=1`.
  - Algorithmic baseline (no LLM): add `--algo`.

## Constraints & Policies

- Model is hard‑locked to `gpt-5-2025-08-07`; do not introduce overrides.
- Use JSON response format and prompts that explicitly include the word “JSON” to comply with model requirements.
- UI should remain minimal and responsive; prefer short stems/options and minimal payloads.
- Do not log or persist PII besides username; keep stats aggregate only.
 - Memory gating: repeats only when not mastered and due by interval; allow a single immediate remediation after a wrong answer.

Optional hardening (Provable Novice Mode):
- `TUTOR_ANONYMIZE=1` → apply per‑user codebook + numeric scrambling to stems/options/rationales.
- `TUTOR_REQUIRE_CITATIONS=1` → credit only if correct AND coverage ≥ τ AND witness re‑pick agrees. Optional `TUTOR_COVERAGE_TAU` (default 0.4).

Adaptivity is lightweight: per‑skill `mastery` (0..1) with 7‑day half‑life decay, misconception counters, and a simple policy: remediation → continue current until mastered → review due → advance when prereqs satisfied.

## Data & Persistence

- Simple JSON store at `data/user_stats.json` via `server/storage.py`.
- Stats shape: per user → `{ per_skill: {skill_id → {correct, wrong, total}}, per_category: {...} }`.
 - Per‑skill fields include `mastery` (0..1), `last_seen_at` (ISO), `misconceptions{}`, and `template_counts{}`.
 - Item aggregates in `items{}` track delivered/correct/wrong, p‑value, flag rate, confidence/time sums, and retirement flag.
- Usernames are stored case‑insensitive and map to a stable `user_id`.

## Skill Map & Categories

- Skills are defined in `docs/rt-psych-tutor/skill_map.psych101.yaml`.
- Category is computed as the top‑level parent of a skill; used to roll up stats.

## Code Pointers

- `server/app.py` — REST endpoints, static UI, prefetch cache, category detection, stats wiring, anonymization and coverage/witness scoring.
- `server/storage.py` — JSON store with user upsert, record, and profile APIs.
- `tutor/llm_openai.py` — OpenAI wrapper; MCQ minimal mode and JSON parsing.
   - Two‑stage self‑verify helper: `answer_mcq(stem, options)`.
 - `tutor/templates.py` — Load curated MCQ templates.
- `web/index.html` — Minimal UI, username memory, and live stats.
   - Tabs for Quiz and Progress; controls for explanations, self‑verify, templates, and modes.

## Extending Safely

- To add richer feedback: add `rationales` back to MCQ prompts and render them in the UI.
- To implement adaptivity: store mastery per skill and schedule review per `docs/rt-psych-tutor/adaptive_algorithms.md`.
- To switch persistence: migrate to Postgres using `docs/rt-psych-tutor/db_schema.sql` and replace `server/storage.py` with DB queries.

## Validation Checklist

- [ ] `.env` loaded and key verified (`tutor.cli verify`).
- [ ] `/api/skills` returns ≥ 1 skill.
- [ ] `/api/generate` returns 5 options and a valid `correct_index`.
 - [ ] Rich generation returns `rationales` and `misconception_tags` arrays aligned to options.
- [ ] Answer recording updates both per‑skill and per‑category totals.
- [ ] (If citations required) `/api/record` returns `credited`, `coverage`, and `witness_pass`.
- [ ] Username persists across sessions and maps to the same `user_id`.
 - [ ] Progress tab shows updated per‑category accuracy and per‑skill mastery bars.

## Troubleshooting

- 400: temperature/JSON format — ensure prompts do not set `temperature` and include “JSON” when using `response_format=json_object`.
- DNS/connectivity — verify outbound HTTPS and that `OPENAI_API_KEY` is present.
- Missing skills — check `docs/rt-psych-tutor/skill_map.psych101.yaml` and file path.
 - If credited=0 while correct>0 — ensure `TUTOR_REQUIRE_CITATIONS` off, or send `citations_text`/`citations[]` with answers and lower `TUTOR_COVERAGE_TAU` for testing.
