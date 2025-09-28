User Model: Mastery, Memory, Misconceptions, Personalization

- Mastery: Per-skill score in [0,1] representing probability of correct recall/application. Initialize low; update after each interaction.
  - BKT-lite update: m' = m + α*(correct - m), where correct ∈ {0,1} (or partial credit), α based on difficulty and answer confidence.
  - Difficulty-adjusted: weight updates more when question difficulty ≥ user ability; cap per-step change.
- Memory Decay: Exponential decay towards baseline b over time Δt since last practice: m(t) = b + (m0 - b)*exp(-λΔt). λ increases for weaker skills, decreases with repeated success.
- Misconceptions: Catalog common misconceptions per skill. Track user_misconception[skill, tag] = evidence score in [0,1]. Increase when answers align with misconception; prioritize remediation when high.
- Speed/Engagement: Track response_time distribution and rapid-guessing flags; adjust difficulty pacing and insert confidence checks as needed.
- Personalization: User prefs/goals (exam date, study pace), prior knowledge bootstrapping, and preferred modalities (text-first, MCQ-first).

State Tracked Per Skill (per user)

- mastery: float
- last_practiced_at: timestamp
- practice_count, success_count
- current_interval: int (days) for spaced repetition bucket
- misconception_scores: map[tag]→float
- recent_response_times: sliding window stats (mean, stdev)

