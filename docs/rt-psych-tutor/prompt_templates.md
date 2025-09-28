Prompt Templates and I/O Specs

Conventions

- All LLM calls request JSON with a fixed schema to simplify parsing.
- System prompts include role, constraints, and grading rubrics.
- Few-shot examples optionally injected for reliability.

1) Generate Multiple-Choice Question (MCQ)

System: You are a psychology tutor generating exam-quality questions aligned to a specified skill. Avoid trivia, prefer conceptual understanding. Include plausible distractors targeting common misconceptions. Difficulty: {difficulty}.

User JSON:
{
  "skill_id": "cog-learning-theories",
  "skill_name": "Learning Theories",
  "bloom": "Understand",
  "constraints": {"num_options": 4, "avoid_repeats": ["...recent stems..."]}
}

Assistant JSON schema:
{
  "question_id": "string",
  "stem": "string",
  "options": ["string", "string", "string", "string"],
  "correct_index": 1,
  "rationales": ["why A", "why B", "why C", "why D"],
  "misconception_tags": ["negative-reinforcement-vs-punishment"],
  "difficulty": "easy|medium|hard"
}

2) Generate Short-Answer Question (SAQ)

Similar system prompt; require concise stem expecting 1–3 sentence answer.

Assistant JSON schema:
{
  "question_id": "string",
  "stem": "string",
  "expected_points": [
    {"key": "definition", "required": true},
    {"key": "example", "required": false}
  ],
  "model_answer": "high-quality reference answer",
  "difficulty": "..."
}

3) Grade Short Answer

System: You grade undergraduate psychology short answers fairly but rigorously. Return scores and evidence. Identify misconceptions.

User JSON:
{
  "stem": "...",
  "expected_points": [{"key": "definition", "required": true}],
  "model_answer": "...",
  "student_answer": "..."
}

Assistant JSON schema:
{
  "score": 0.0,             // 0..1
  "correct": true,          // thresholded from score
  "found_points": ["definition"],
  "missing_points": ["example"],
  "misconception_tags": ["..."],
  "feedback": "2–3 sentences: what was right, what to fix"
}

4) Feedback and Remediation

System: Give brief, supportive feedback. If a misconception is detected, explain the contrast and ask a targeted follow-up.

Assistant JSON schema:
{
  "explanation": "succinct explanation tied to the skill",
  "contrast_example": "if relevant",
  "follow_up_question": {
    "type": "mcq|saq",
    "payload": { /* question schema as above */ }
  }
}

5) Next Question Proposal (optional)

System: Propose the next best question given user state and skill graph; prefer mixing ~30% spaced review.

User JSON includes summary of recent performance and due reviews; assistant returns target skill_id + difficulty.

