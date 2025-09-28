Adaptive Algorithms

Next Question Selection (per interaction)

Inputs: current_skill, user model U, due_reviews, fatigue/engagement flags

1) If remediation_needed(U, current_skill): return remediation(current_skill)
2) If review_due(due_reviews) and mix_ratio unmet: return pick_review(due_reviews)
3) If not mastered(current_skill): return continue_current(current_skill)
4) Else: return advance_to(next_skill_by_prereq(current_skill, U))

Difficulty Adaptation

- Maintain user ability θ per domain; question difficulty d ∈ [easy, hard].
- Target d ≈ θ for practice; increase d after streaks of correct/fast answers; decrease after errors or slow responses.
- For MCQ, increase distractor plausibility and reduce cues as d rises; for SAQ, require more precise/abstract reasoning.

Spaced Repetition Scheduling (bucketed)

Buckets: [1, 3, 7, 14, 30] days. Each skill has current_interval.

On answer(correct):
- If correct with high confidence and normal speed: promote to next longer bucket
- If borderline: keep same bucket
- If incorrect/slow: demote to shorter bucket and schedule remediation now/later in session

Pseudo-Implementation

function schedule_next(skill, result, U):
  ivals = [1,3,7,14,30]
  i = index_of(U.current_interval[skill]) or 0
  if result.correct and result.confidence_high and result.time_ok:
    i = min(i+1, len(ivals)-1)
  elif result.correct:
    i = i
  else:
    i = max(i-1, 0)
  U.current_interval[skill] = ivals[i]
  U.next_review_at[skill] = now + days(ivals[i])

Remediation Strategy

- Identify misconception tags from grading; ask a targeted follow-up that contrasts correct vs misconception.
- Provide brief explanation linking to definition and example; re-ask similar item to confirm fix.

