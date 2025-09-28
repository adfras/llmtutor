# Software Requirements Specification (SRS) — Real‑Time Psych Tutor

Version: 1.0
Status: Active (living)
Owners: Product, Engineering, Learning Science

## 1. Introduction
- Purpose: Specify functional and non‑functional requirements for a minimal, real‑time, LLM‑powered psychology tutor that generates MCQs, tracks learning, and adapts using mastery + memory decay.
- Scope: Web UI (single page), FastAPI backend, OpenAI model integration, JSON storage. Out of scope: full LMS integration, grading beyond SAQ prototype, accounts beyond username.

## 2. Stakeholders
- Learners: Students using the UI.
- Content Owners: Educators curating templates and OER links.
- Engineers: Building/operating the service.

## 3. Constraints
- Fixed model: `gpt-5-nano-2025-08-07` (no override).
- Strict JSON outputs; prompts must say “JSON”.
- Privacy: Only username stored; aggregate stats only.
- Minimal UI: no heavy frameworks.

## 4. Definitions
- Skill: Node in PSY101 skill map; category is top‑level ancestor.
- Mastery: Per‑skill score in [0,1] with 7‑day half‑life decay.
- Due: A skill is due when elapsed time ≥ interval(mastery) [1/3/7/14 days].
- Remediation: One immediate follow‑up after a wrong answer on the same skill.

## 5. System Context
- UI calls REST endpoints on backend; backend calls OpenAI; storage persists JSON file.
- See STS for authoritative API contracts.

## 6. Functional Requirements

FR‑1 Skills
- FR‑1.1 The system shall provide `GET /api/skills` returning skills with `id,name,bloom`.
- FR‑1.2 The system should sort skills by `name` (case‑insensitive) for usability.

FR‑2 Generation
- FR‑2.1 The system shall generate MCQs with minimal payload `{stem,options[],correct_index}` and attach a deterministic `item_id`.
- FR‑2.2 When `rich=true`, it shall include `rationales[]`, `misconception_tags[]`, and `prompt_id`.
- FR‑2.3 Optional: `verify=true` triggers a second model pass that answers the item; on mismatch, regenerate once.
- FR‑2.4 Optional: `use_templates=true` steers generation with curated templates; when `user_id` provided, select the least‑used template for that user+skill; attach `template_id`.
- FR‑2.5 The system shall run rule checks on generated MCQs and retry once on failure.

FR‑3 Recording
- FR‑3.1 The system shall record `{user_id,skill_id,correct}` and update per‑skill/per‑category totals atomically.
- FR‑3.2 The system shall update per‑skill `mastery` using EMA and time decay; update `last_seen_at` to now.
- FR‑3.3 When provided, the system shall record `misconception_tag`, `item_id`, `confidence (1..5)`, `time_to_answer_ms`, and `template_id`.
- FR‑3.4 The system shall aggregate per‑item totals and compute derived metrics (`p_value`, `flag_rate`) and mark `retired=true` when thresholds are exceeded.

FR‑4 Adaptivity (Next Item Selection)
- FR‑4.1 If last answer was wrong and `current_skill_id` present, the next item is remediation on the same skill (single immediate follow‑up).
- FR‑4.2 Otherwise, continue current only if `decayed_mastery < 0.8` AND the skill is due by interval (1/3/7/14 days by mastery band).
- FR‑4.3 Otherwise, if any previously practiced skills are due, select the one with the lowest decayed mastery; break ties by longest elapsed time since last_seen.
- FR‑4.4 Otherwise, advance to a new skill whose prerequisites’ decayed mastery ≥ 0.8; prefer siblings of the current skill; otherwise any eligible.

FR‑5 Item Validity & Analytics
- FR‑5.1 Rule checks: non‑trivial stem (≥3 words), no duplicate options, no “All/None of the above/these”, valid `correct_index`, and avoid extreme option length imbalance (max length ≤ 3×mean).
- FR‑5.2 The system shall attach `item_id` as a SHA‑256 fingerprint of skill_id, stem, options, and correct_index.
- FR‑5.3 The system shall store and expose per‑item aggregates (delivered, correct, wrong, flags) and confidence/time sums; compute `p_value = correct/delivered` and `flag_rate = flags/delivered`.
- FR‑5.4 Retirement: after `delivered ≥ 100`, set `retired=true` if `flag_rate > 0.01` or `p_value < 0.1` or `p_value > 0.9`.

FR‑6 OER & Reporting
- FR‑6.1 The system shall provide `GET /api/skill/{skill_id}/oer` with `{title,url,snippet}`.
- FR‑6.2 The system shall provide `GET /api/dashboard` with `{ top_missed_skills[], flagged_items[] }` for quick QA.

FR‑7 UI Behavior
- FR‑7.1 The UI shall offer toggles: Show explanations (default on), Self‑verify, Use templates, and a Mode selector (Practice/Check‑up).
- FR‑7.2 Practice mode shows immediate feedback; Check‑up mode hides feedback during a 10‑item block and shows a summary at block end.
- FR‑7.3 The UI shall provide a “Progress” tab with: total answered & accuracy; per‑category accuracy bars; top skills with mastery bars.
- FR‑7.4 The UI shall include a 1–9 keyboard select and Enter=Next; options shall be accessible (role=button, focusable); outcome region shall be `aria-live`.
- FR‑7.5 The UI shall include a “Report a problem” action that calls `/api/item/flag` with the current `item_id`.

## 7. Detailed Use Cases

UC‑1 Practice a skill
- Preconditions: User has a `user_id`.
- Main Flow: User clicks Generate → Server returns MCQ → User answers → Server records correctness, mastery updates; UI shows feedback and rationale.
- Alternate: Self‑verify on; server regenerates once if mismatch.

UC‑2 Adaptive next
- Preconditions: UC‑1 at least once.
- Main Flow: UI calls `/api/next` with `last_correct` → Server selects skill per FR‑4 → Returns MCQ.

UC‑3 Check‑up block
- Main Flow: User selects Check‑up mode → Completes 10 items → UI presents mini summary; server has recorded items individually.

UC‑4 Flag item
- Main Flow: User clicks Report → `/api/item/flag` increments flag counter.

UC‑5 Learn more
- Main Flow: UI calls `/api/skill/{id}/oer` and opens link.

## 8. Data Dictionary

- mastery (float): [0,1] probability proxy the learner answers correctly now (decayed EMA).
- p_value (float): proportion correct over delivered for an item.
- flag_rate (float): flags/delivered for an item.
- due (bool): elapsed_days ≥ review_interval(mastery).

## 9. Non‑Functional Requirements
- Latency: minimal P95 ≤ 2.5s; rich+verify P95 ≤ 4.0s.
- Availability: 99.5%/month; 5xx ≤ 1%/24h.
- Privacy: only username persisted; no free‑text answers stored.
- Accessibility: keyboard operation; role=button on options; `aria-live` status.

## 10. Security & Threats
- Secrets in env/.env; never log keys.
- Validate inputs; enforce known `skill_id`.
- CORS restricted in production.
- Abuse control (future): per‑IP rate limiting on write endpoints.

## 11. Acceptance Criteria (Expanded)
- AC‑1 `/api/skills` returns sorted list with ≥1 entry.
- AC‑2 `/api/generate` minimal returns 5 options and valid index; rich adds rationales/tags of same length and `item_id`.
- AC‑3 `/api/record` increments per‑skill/per‑category totals, updates `mastery` and `last_seen_at`; accepts optional fields without error.
- AC‑4 `/api/next` adheres to FR‑4 decision policy under testable scenarios (wrong→remediation; not‑due current→advance/review; due review with lowest mastery selected).
- AC‑5 Validator rejects duplicates/All‑None/imbalanced; retry occurs once; `item_id` attached.
- AC‑6 UI controls present; keyboard shortcuts work; Progress tab renders bars from live stats.

## 12. Open Issues
- Discrimination metric (point‑biserial) requires cohort/block context; placeholder for future.
- Template coverage completeness across all skills pending curation.

## 7. Data Requirements
- Skill map stored in YAML (`docs/rt-psych-tutor/skill_map.psych101.yaml`).
- User stats JSON at `data/user_stats.json` with shape defined in STS (users, usernames, stats, items).
- OER link map at `docs/rt-psych-tutor/oer_links.yaml`.
- Optional templates at `docs/rt-psych-tutor/mcq_templates.yaml`.

## 8. Non‑Functional Requirements
- Latency: MCQ minimal P95 ≤ 2.5s; with `verify=true`, P95 ≤ 4s.
- Availability: 99.5% monthly (best effort in MVP).
- Privacy: only username persisted; no free‑text answers stored.
- Accessibility: keyboard operation; roles and `aria-live` for result updates.

## 9. Security & Privacy
- Load `OPENAI_API_KEY` from env/.env; never log.
- No PII besides username; aggregate stats only.
- CORS restricted in production deployments.

## 10. Acceptance Criteria
- AC‑1 `/api/skills` returns ≥1 skill.
- AC‑2 Minimal `/api/generate` returns 5 options with valid index; Rich returns rationales + tags of equal length.
- AC‑3 `/api/record` increments per‑skill and per‑category totals and updates mastery/last_seen_at; optional fields are ingested when present.
- AC‑4 `/api/next` follows remediation → continue‑if‑due → review‑due → advance policy.
- AC‑5 Generated MCQs pass rule validator or are retried once; every item has an `item_id`.
- AC‑6 UI shows explanations by default, toggles work, keyboard shortcuts work, and the Progress tab renders bars from live stats.

## 11. Future Work (Not in scope of MVP)
- True discrimination analysis (point‑biserial) with cohort/block context.
- Educator dashboard with item review workflows and retirement controls.
- Export/import of anonymized interaction logs.

Links
- STS (technical standard): `docs/rt-psych-tutor/STS.md`
- Architecture: `docs/rt-psych-tutor/architecture.md`
