# Real-Time LLM‑Powered Psychology Tutor

Minimal, fast, and practical demo of a real‑time tutor for Intro Psychology. It generates multiple‑choice questions on the fly using an LLM, provides a one‑click web UI, and persists per‑user correctness stats by skill and by category across sessions.

## Highlights

- Minimal web UI: generate MCQs with 5 options; immediate feedback.
- Fixed model: OpenAI `gpt-5-nano-2025-08-07` for generation (no overrides).
- Persistent stats: per‑skill and per‑category in `data/user_stats.json`.
- Provable novice mode (optional): anonymization + evidence‑gated scoring to prevent pretraining short‑cuts and force learning from NOTES.
- Design pack: architecture, skill map, prompts, API, DB schema in `docs/rt-psych-tutor/`.
- STS/SRS: contracts and acceptance in `docs/rt-psych-tutor/STS.md` and `SRS.md`.
- Progress tab: live per‑category accuracy and per‑skill mastery bars.

## Repo Layout

- `server/app.py` – FastAPI server (REST + static UI)
- `server/storage.py` – Simple JSON store for user stats and usernames
- `web/index.html` – Minimal UI (no build tools)
- `tutor/` – CLI + OpenAI wrapper + skill map loader
- `docs/rt-psych-tutor/` – System design artifacts (architecture, skill map, prompts, API)
- `requirements.txt` – Python dependencies

## Synthetic Student (DeepInfra / OpenAI)

Use the included harness to simulate a “student” answering items so you can test adaptivity and analytics end‑to‑end.

Prereqs:
- Server running (see Quickstart). Tutor generation model is fixed.
- `.env` contains `DEEPINFRA_API_KEY` (for open‑source models) and `OPENAI_API_KEY`.

Common runs:

```
export $(grep -v '^#' .env | xargs)
# Open‑source (DeepInfra) student
python student_bot_deepinfra.py --username os_llama --n 100 --fast --provider deepinfra --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
python student_bot_deepinfra.py --username os_qwen  --n 100 --fast --provider deepinfra --model Qwen/Qwen2.5-7B-Instruct

# Closed‑book novice (uses only accumulated NOTES; good for learning curves)
TUTOR_ANONYMIZE=1 uvicorn server.app:app --reload &  # or set in your shell
python student_bot_deepinfra.py --username novice_qwen --n 100 --fast --provider deepinfra --model Qwen/Qwen2.5-7B-Instruct --closed-book --notes-file data/notes_novice_qwen.txt

# Algorithmic student (no LLM; NOTES‑overlap baseline)
python student_bot_deepinfra.py --username algo1 --n 100 --fast --algo --closed-book --notes-file data/notes_algo1.txt
```

Notes:
- Harness supports DeepInfra (`--provider deepinfra`) and OpenAI (`--provider openai`).
- Closed‑book mode only uses the NOTES the bot accumulates from tutor rationales.
- Algorithmic mode provides a true “no prior knowledge” baseline with simple NOTES→option overlap.

## Prerequisites

- Python 3.10+
- OpenAI API key in `.env` at repo root, e.g.:

```
OPENAI_API_KEY=sk-...
```

Notes:
- The project uses the fixed model `gpt-5-nano-2025-08-07`. Changing the model requires code changes.

## Quickstart (Web UI)

1) Create a venv and install deps

```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run the server

```
uvicorn server.app:app --reload
```

3) Open the UI

- http://localhost:8000
- Enter a username (remembered across sessions)
- Click “Generate Question”, then pick an option to see Correct/Incorrect
- Running totals show for the current skill and its category
 - Try the toggles: Self‑verify (more accurate, slower), Use templates (curated patterns), and Mode (Practice vs Check‑up).
 - Check the Progress tab for per‑category accuracy and per‑skill mastery bars.

## API Endpoints

- `GET /api/skills` → list available skills (from the skill map)
- `POST /api/generate` → generate a question
  - Body: `{ "skill_id": "cog-learning-theories", "type": "mcq", "difficulty": "medium", "num_options": 5, "rich"?: true, "verify"?: true, "use_templates"?: true, "user_id"?: "..." }`
  - Returns: `{ type, skill_id, question: { stem, options[], correct_index } }`
- `POST /api/record` → record answer outcome
  - Body (basic): `{ user_id, skill_id, correct, item_id?, confidence?, time_to_answer_ms?, misconception_tag? }`
  - Body (evidence‑gated, optional): include `stem`, `options[]`, `correct_index`, optional `rationales[]`, and `citations[]`/`citations_text` so the server can compute coverage and run a witness check.
- `GET /api/user/{user_id}/stats` → `{ per_skill, per_category }`
- `POST /api/user/upsert` → `{ user_id, username }` (idempotent by username)
- `GET /api/user/by-name/{username}` → `{ user_id, username, stats }`
- `POST /api/next` → adaptive next question
  - Body: `{ user_id, current_skill_id?, last_correct?, type:"mcq", difficulty, num_options, rich?, verify?, use_templates? }`
  - Returns: `{ type, skill_id, reason, question }`
- `POST /api/item/flag` → increment flag count for a problematic item
  - Body: `{ item_id: "sha256:..." }`
  - Returns: `{ ok: true, item: { item_id, flags } }`
- `GET /api/skill/{skill_id}/oer` → returns `{ skill_id, title, url, snippet }` for a “Learn more” link
- `GET /api/dashboard` → admin snapshot `{ top_missed_skills[], flagged_items[] }`

## CLI (Optional)

- Verify key/model: `python -m tutor.cli verify`
- Generate MCQ: `python -m tutor.cli generate-mcq --skill-id cog-learning-theories`
- Generate SAQ: `python -m tutor.cli generate-saq --skill-id cog-memory`
- Grade SAQ: `python -m tutor.cli grade-saq --stem "..." --expected-points "[...]" --model-answer "..." --student-answer "..."`

## Skill Map

- Source: `docs/rt-psych-tutor/skill_map.psych101.yaml`
- Each skill has `id`, `name`, optional `parent`, and a Bloom level.
- The UI defaults to `cog-learning-theories` but the API accepts any listed `skill_id`.

## Performance Notes

- Minimal MCQ payload (stem, options, correct_index) keeps responses small.
- The server reuses a single OpenAI client and prefetches the next question per (skill, difficulty, num_options).
- Mastery uses a simple EMA plus 7‑day half‑life decay; review injection is based on mastery bands (1/3/7/14 days).

## Data & Privacy

- User profiles and stats persist to `data/user_stats.json`.
- Only username and aggregate counts are stored; no free‑text answers are kept by default.
- See `docs/rt-psych-tutor/privacy_security.md` for design considerations.

## Adaptivity

- Per‑skill state now includes `mastery` (0..1), `last_seen_at` (ISO time), and `misconceptions` (tag→count).
- `POST /api/record` updates mastery with decay+EMA and optionally increments a `misconception_tag` when wrong.
- `POST /api/next` follows policy: remediation → continue current if not mastered → review‑due → advance to next skill with prereqs satisfied.

## Provable Novice Mode (Optional)

These hardening features structurally prevent a small OS LLM “student” from using pretraining to answer and require evidence use:
- Anonymization: per‑user codebook + numeric scrambler applied to stems/options/rationales (`TUTOR_ANONYMIZE=1`).
- Closed‑book student: uses only accumulated NOTES (correct option + rationale lines) when answering.
- Evidence‑gated scoring: enable `TUTOR_REQUIRE_CITATIONS=1` to credit only answers that are correct AND sufficiently covered by cited NOTES/OER AND pass a witness re‑pick check.

See `Provable_Novice_Learning_Blueprint.docx` for the hardened blueprint.

## Rich Explanations (Optional)

- Pass `"rich": true` when generating to include `rationales` and `misconception_tags` aligned to options. The web UI has a “Show explanations” toggle.
  - Explanations are enabled by default in the UI; toggle remains to disable. A “Self‑verify” option runs a second LLM pass to double‑check the key.

## Templates & Coverage Rotation

- Optional template‑driven generation rotates across curated templates per skill (least‑used first per user). Define templates in `docs/rt-psych-tutor/mcq_templates.yaml`.

## Study Modes

- Practice: immediate feedback and rationales.
- Check‑up: delayed feedback after a 10‑item block with a mini summary (client‑side, storage still records each response).

## Validity Guardrails

- The server runs rule‑based validation on generated MCQs (single key, no duplicates, no All/None, balanced lengths, non‑trivial stem) and retries once if validation fails.
- Each item receives a deterministic `item_id` fingerprint (`sha256`) from stem/options/correct_index/skill.

## Extending

- Show richer feedback: enable rationales and explanations in `tutor/llm_openai.py` by disabling minimal mode.
- Adaptive scheduling: wire `docs/rt-psych-tutor/adaptive_algorithms.md` and persist mastery per skill.
- DB backend: replace JSON storage with the Postgres schema in `docs/rt-psych-tutor/db_schema.sql`.

## Docs

- Full design pack in `docs/rt-psych-tutor/`:
  - Architecture, skill map, user model, adaptive logic, prompts, DB schema, API, privacy/security
  - STS: `docs/rt-psych-tutor/STS.md` (technical standards and contracts)
  - SRS: `docs/rt-psych-tutor/SRS.md` (requirements & acceptance criteria)
