System Architecture Overview

- Frontend: Web/mobile UI (chat or quiz) for login, question display, answers, and feedback. Communicates via REST for standard calls and WebSocket for real-time streaming/latency-sensitive updates.
- Backend: Orchestrates tutoring logic; formats prompts; calls LLM; updates mastery; schedules next questions; exposes API; runs async jobs for heavier tasks; uses cache for hot data.
- LLM Integration: Provider/API or local model. Roles: generate questions, grade short answers, produce explanations, surface misconceptions, and propose next-step hints.
- Database: Stores users, skills/topic graph, mastery, interactions/logs, misconceptions, and sessions. Optimized indexes for per-user, per-skill lookups.
- Cache/Queue: Redis for recent Q&A and idempotency; Celery/worker queue for batch generation, analytics, and spaced repetition recalcs.
- Observability: Structured logs, metrics (latency, token use, answer rates), and alerts.

Request Flow (Happy Path)

1) UI submits answer → Backend validates + timestamps
2) If MCQ: check locally. If short answer: grade via LLM
3) Backend updates user model (mastery, decay reset, misconceptions)
4) Backend decides what’s next (adaptive selection) and prompts LLM to generate the next question
5) Returns feedback + next question to UI (stream when possible)
6) Async tasks update aggregates and schedule reviews

Performance Notes

- Prefer small/fast models for grading/simple prompts; reserve higher-capacity models for generation requiring nuance.
- Cache explanation re-uses; keep dynamic, personalized generation uncached.
- Use streaming responses and optimistic UI to reduce perceived latency.

