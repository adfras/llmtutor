from __future__ import annotations
import os
from typing import Dict, Any, List

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


TEMPLATES_PATH = os.path.join("docs", "rt-psych-tutor", "mcq_templates.yaml")


def load_mcq_templates(path: str = TEMPLATES_PATH) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Please `pip install -r requirements.txt`.")
    if not os.path.exists(path):
        return {"skills": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    skills = data.get("skills", {}) or {}
    return {"skills": skills}


def templates_for_skill(skill_id: str, templates: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    templates = templates or load_mcq_templates()
    skill_templates = templates.get("skills", {}).get(skill_id, {}).get("templates", []) or []
    return skill_templates

