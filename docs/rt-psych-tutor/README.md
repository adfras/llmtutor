Real-Time LLM-Powered Psychology Tutor — Design Pack

This folder implements the system design described in “Real-Time LLM-Powered Psychology Tutor: System Design Guide”. It provides concrete, implementation-ready artifacts (no app code) covering architecture, curriculum structure, adaptive logic, data model, API, prompts, and privacy.

- Architecture: system components and request flow (architecture.md)
- Skill Map: PSY 101 topic/skill graph (skill_map.psych101.yaml)
- User Model: mastery, decay, misconceptions, personalization (user_model.md)
- Adaptive Logic: selection, difficulty, scheduling, remediation (adaptive_algorithms.md)
- Prompts: generation, grading, feedback templates with I/O specs (prompt_templates.md)
- Database: normalized Postgres schema (db_schema.sql)
- API: OpenAPI for core tutoring endpoints (api_spec.yaml)
- Privacy & Security: data handling and compliance (privacy_security.md)
- CLI Prototype: OpenAI-backed generator/grader (see “Usage” below)
 - STS: Software Technical Standard (STS.md)
 - SRS: Software Requirements Specification (SRS.md)

Provable Novice Mode (Implementation Notes)

This repo includes an optional “provable novice” path that enforces learning from session evidence and blocks pretraining short‑cuts:
- Anonymization: per‑user codebook and numeric scrambler for stems/options/rationales (enable with `TUTOR_ANONYMIZE=1`).
- Closed‑book student: the synthetic student only uses accumulated NOTES (correct option + rationale) to answer; no raw domain terms.
- Evidence‑gated scoring: when `TUTOR_REQUIRE_CITATIONS=1`, the server credits answers only if (a) correct, (b) coverage ≥ τ between student citations and the gold explanation, and (c) a witness re‑pick using only the citations matches the student’s index. Configure τ with `TUTOR_COVERAGE_TAU`.

See `Provable_Novice_Learning_Blueprint.docx` for the hardened blueprint and rationale.

Suggested starting points:
- Begin with the skill map and import it into the DB.
- Use the prompts and adaptive pseudocode to drive a prototype backend.
- Wire the API to the DB and LLM provider of your choice.
 - Use SRS as the acceptance checklist; use STS for contracts and NFRs.

Usage (CLI Tutor Prototype)

- Install deps: `pip install -r requirements.txt`
- Ensure `.env` contains `OPENAI_API_KEY=...`
- Note: The CLI uses fixed model `gpt-5-nano-2025-08-07`.
- Generate an MCQ: `python -m tutor.cli generate-mcq --skill-id cog-learning-theories`
- Generate an SAQ: `python -m tutor.cli generate-saq --skill-id cog-memory`
- Grade an SAQ:
  `python -m tutor.cli grade-saq --stem "Define classical conditioning" --expected-points "[{\"key\":\"definition\",\"required\":true}]" --model-answer "Association learning with neutral and unconditioned stimuli" --student-answer "Learning by pairing a neutral stimulus with one that naturally elicits a response"`
- Quick demo: `python -m tutor.cli demo`

Synthetic Student (Novice Simulation)

```
# Start server with anonymization and (optionally) citation gating
TUTOR_ANONYMIZE=1 TUTOR_REQUIRE_CITATIONS=1 uvicorn server.app:app --reload

# Open‑source student (DeepInfra): closed-book novice
export $(grep -v '^#' .env | xargs)  # needs DEEPINFRA_API_KEY
python ../../student_bot_deepinfra.py --username novice --n 100 --fast \
  --provider deepinfra --model Qwen/Qwen2.5-7B-Instruct --closed-book \
  --notes-file ../../data/notes_novice.txt

# Algorithmic baseline (no LLM)
python ../../student_bot_deepinfra.py --username algo --n 100 --fast --algo --closed-book \
  --notes-file ../../data/notes_algo.txt
```
