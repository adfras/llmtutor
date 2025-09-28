from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from tutor.utils import extract_json


MODEL_NAME = "gpt-5-nano-2025-08-07"


class OpenAILLM:
    def __init__(self):
        # Load env
        from tutor.utils import load_env_dotenv_fallback
        load_env_dotenv_fallback()
        self.model = MODEL_NAME
        self._mock = os.getenv("TUTOR_MOCK_LLM") == "1"
        if self._mock:
            self.client = None  # type: ignore
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
            # Lazy import to avoid hard dependency when listing help
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)

    def _chat_json(self, system: str, user: str) -> Dict[str, Any]:
        # Prefer JSON mode when available
        if self._mock:
            # Return a minimal echo-like JSON for testing
            try:
                return extract_json(user)
            except Exception:
                return {"ok": True}
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return extract_json(content)
        except Exception:
            # Fallback to plain JSON parse
            return extract_json(content)

    def verify_key_and_model(self) -> Dict[str, Any]:
        """Verify API key works and model is callable."""
        info: Dict[str, Any] = {"model": self.model}
        if self._mock:
            info.update({"models_list_ok": True, "chat_ok": True, "mock": True})
            return info
        # Check models endpoint
        try:
            _ = self.client.models.list()
            info["models_list_ok"] = True
        except Exception as e:
            info["models_list_ok"] = False
            info["models_error"] = str(e)
        # Try a minimal chat call
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return exactly the provided JSON."},
                    {"role": "user", "content": '{"ok": true}'}
                ],
            )
            text = resp.choices[0].message.content or "{}"
            extract_json(text)  # validate JSON parsable
            info["chat_ok"] = True
        except Exception as e:
            info["chat_ok"] = False
            info["chat_error"] = str(e)
        return info

    # MCQ generation
    def generate_mcq(
        self,
        skill: Dict[str, Any],
        difficulty: str = "medium",
        avoid_repeats: List[str] | None = None,
        num_options: int = 5,
        minimal: bool = True,
        template: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import json, random
        avoid_repeats = avoid_repeats or []
        if self._mock:
            # Deterministic pseudo-MCQ for fast simulations
            sid = skill["id"]; sname = skill["name"]
            opts = [f"Option {chr(65+i)}" for i in range(num_options)]
            correct = random.randrange(num_options)
            if minimal:
                return {"stem": f"Which relates to {sname}?", "options": opts, "correct_index": correct}
            else:
                r = [f"Reason for {o}" for o in opts]
                mt = ["" for _ in opts]
                return {
                    "question_id": f"mock-{sid}-{random.randint(1,999999)}",
                    "stem": f"Best example of {sname}?",
                    "options": opts,
                    "correct_index": correct,
                    "rationales": r,
                    "misconception_tags": mt,
                    "difficulty": difficulty,
                }
        if minimal:
            system = (
                "You are a psychology tutor generating a concise multiple-choice question. "
                "Target the given skill, avoid trivia, prefer conceptual understanding. "
                "Return only a JSON object with keys: stem (string), options (array, length = constraints.num_options), correct_index (int). "
                "Keep stem under 25 words. Keep each option under 8 words. No explanations, no rationales, no extra fields."
            )
        else:
            system = (
                "You are a psychology tutor generating exam-quality multiple-choice questions. "
                "Target the given skill, avoid trivia, prefer conceptual understanding. "
                "Include plausible distractors that reflect common misconceptions. "
                "Return only a JSON object with keys: question_id (string), stem (string), options (array of strings with length equal to constraints.num_options), correct_index (int), rationales (array of strings matching options length), misconception_tags (array of strings, same length as options), difficulty (string)."
            )
        payload = {
            "skill_id": skill["id"],
            "skill_name": skill["name"],
            "bloom": skill.get("bloom", ""),
            "difficulty": difficulty,
            "constraints": {"num_options": num_options, "avoid_repeats": avoid_repeats},
        }
        if template:
            payload["template_id"] = template.get("id")
            payload["template"] = template.get("description")
            if template.get("misconception_tags"):
                payload["misconception_tags"] = template["misconception_tags"]
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json(system, user)

    # SAQ generation
    def generate_saq(self, skill: Dict[str, Any], difficulty: str = "medium") -> Dict[str, Any]:
        import json
        if self._mock:
            return {
                "question_id": f"mock-saq-{skill['id']}",
                "stem": f"Briefly define {skill['name']}",
                "expected_points": [{"key": skill["name"], "required": True}],
                "model_answer": f"Definition of {skill['name']}.",
                "difficulty": difficulty,
            }
        system = (
            "You are a psychology tutor generating a short-answer question. "
            "Expect a concise 1–3 sentence answer. Provide expected points and a reference answer. "
            "Return only a JSON object with keys: question_id (string), stem (string), expected_points (array of {key, required}), model_answer (string), difficulty (string)."
        )
        payload = {
            "skill_id": skill["id"],
            "skill_name": skill["name"],
            "bloom": skill.get("bloom", ""),
            "difficulty": difficulty,
        }
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json(system, user)

    # Grade SAQ
    def grade_saq(self, stem: str, expected_points: List[Dict[str, Any]], model_answer: str, student_answer: str) -> Dict[str, Any]:
        if self._mock:
            # Naive match: correct if any expected key appears
            text = (student_answer or '').lower()
            found = [p["key"] for p in expected_points if p["key"].lower() in text]
            correct = bool(found)
            return {"score": 1.0 if correct else 0.0, "correct": correct, "found_points": found, "missing_points": [], "misconception_tags": [], "feedback": "Mock grading."}
        system = (
            "You grade undergraduate psychology short answers fairly and rigorously. "
            "Return a JSON object with numeric score 0..1, boolean correct, found_points, missing_points, misconception_tags, and 2–3 sentence feedback."
        )
        import json

        user_obj = {
            "stem": stem,
            "expected_points": expected_points,
            "model_answer": model_answer,
            "student_answer": student_answer,
        }
        user = json.dumps(user_obj, ensure_ascii=False)
        return self._chat_json(system, user)

    # Self-verify MCQ: model answers the item independently
    def answer_mcq(self, stem: str, options: List[str]) -> Dict[str, Any]:
        import json, random
        if self._mock:
            return {"chosen_index": 0 if options else 0, "confidence": 0.5}
        system = (
            "You are answering a multiple-choice question. Return only JSON with keys: chosen_index (int), confidence (0..1)."
        )
        payload = {"stem": stem, "options": options}
        user = json.dumps(payload, ensure_ascii=False)
        return self._chat_json(system, user)
