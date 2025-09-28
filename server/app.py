from __future__ import annotations
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tutor.skill_map import load_skill_map
from tutor.llm_openai import OpenAILLM
from tutor.templates import templates_for_skill
from tutor.utils import load_env_dotenv_fallback
from .storage import StatsStore


load_env_dotenv_fallback()
app = FastAPI(title="Psych Tutor UI Server")

# Global LLM instance and simple prefetch cache
LLM = OpenAILLM()
STORE = StatsStore()
from typing import Tuple, List
import threading

_CACHE: dict[Tuple[str, str, int], List[dict]] = {}
_LOCK = threading.Lock()

# -------- Anonymization helpers (provable novice blueprint: codebook + numeric scrambler) --------
import re as _re

# Curated high-signal psychology terms/phrases to anonymize (non-exhaustive)
_PSY_TERMS = {
    # General
    "memory","working memory","short-term memory","long-term memory","attention","language","reasoning",
    "learning","conditioning","classical conditioning","operant conditioning","reinforcement","punishment","extinction",
    "stimulus","response","unconditioned stimulus","conditioned stimulus","unconditioned response","conditioned response",
    # Research methods
    "hypothesis","variable","independent variable","dependent variable","control group","random assignment","validity","reliability",
    # Biopsychology
    "neuron","dendrite","axon","synapse","neurotransmitter","dopamine","serotonin","acetylcholine","gaba","glutamate",
    "amygdala","hippocampus","thalamus","hypothalamus","prefrontal cortex","cortex","limbic system","brainstem",
    # Developmental
    "piaget","sensorimotor","preoperational","concrete operational","formal operational","conservation","attachment","strange situation",
    # Social
    "attitude","persuasion","cognitive dissonance","conformity","obedience","asch","milgram","prejudice","stereotype",
    # Personality
    "big five","trait","introversion","extraversion","neuroticism","agreeableness","conscientiousness","openness",
    # Abnormal
    "anxiety","phobia","panic","depression","bipolar","schizophrenia","psychosis","cognitive behavioral therapy","cbt",
}

def _vocab_tokens(smap) -> list[str]:
    # Extract token/phrase vocab from skill names plus curated terms
    vocab: set[str] = set()
    for s in smap["skills"].values():
        name = (s.get("name") or "").lower()
        for tok in _re.findall(r"[a-zA-Z]{4,}", name):
            vocab.add(tok)
    # Include curated phrases/terms
    vocab.update(_PSY_TERMS)
    # Common stop words to drop
    stop = {"psychology","introductory","understand","analysis","apply","create","evaluate","development","social","personality","abnormal","research","methods"}
    return [t for t in vocab if t not in stop]

def _apply_codebook(text: str, codebook: dict[str,str]) -> str:
    # Replace whole-word, case-insensitive
    def repl(m):
        w = m.group(0)
        k = w.lower()
        ct = codebook.get(k)
        return ct or w
    if not codebook:
        return text
    # Sort by length to replace longer phrases first
    keys = sorted((k for k in codebook.keys() if k), key=len, reverse=True)
    if not keys:
        return text
    pattern = _re.compile(r"\b(" + "|".join(_re.escape(k) for k in keys) + r")\b", _re.I)
    try:
        return _re.sub(pattern, repl, text)
    except Exception:
        return text

def _scramble_numbers(text: str, a: int, b: int) -> str:
    def repl(m):
        try:
            x = int(m.group(0))
            y = a * x + b
            return str(y)
        except Exception:
            return m.group(0)
    return _re.sub(r"\b\d{1,4}\b", repl, text)

def _anonymize_question_for_user(q: dict, user_id: str, smap) -> dict:
    if not user_id:
        return q
    vocab = _vocab_tokens(smap)
    codebook = STORE.get_or_create_codebook(user_id, vocab)
    nk = STORE.get_or_create_numeric_key(user_id)
    # Extend codebook with tokens from this question (length>=5, alpha)
    toks: list[str] = []
    try:
        texts = []
        if isinstance(q.get("stem"), str):
            texts.append(q.get("stem") or "")
        if isinstance(q.get("options"), list):
            texts.extend([o or "" for o in q.get("options")])
        raw = "\n".join(texts).lower()
        toks = list(set(_re.findall(r"[a-zA-Z]{5,}", raw)))
    except Exception:
        toks = []
    if toks:
        codebook = STORE.extend_codebook(user_id, toks)
    def transform(s: str) -> str:
        s2 = _apply_codebook(s or "", codebook)
        s3 = _scramble_numbers(s2, nk["a"], nk["b"])
        return s3
    out = dict(q)
    if isinstance(out.get("stem"), str):
        out["stem"] = transform(out.get("stem") or "")
    if isinstance(out.get("options"), list):
        out["options"] = [transform(o or "") for o in (out.get("options") or [])]
    if isinstance(out.get("rationales"), list):
        out["rationales"] = [transform(r or "") for r in (out.get("rationales") or [])]
    out["anonymized"] = True
    return out

def _prefetch(key: Tuple[str, str, int], smap):
    try:
        skill_id, difficulty, num_options = key
        skill = smap["skills"].get(skill_id) or next(iter(smap["skills"].values()))
        q = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=True)
        with _LOCK:
            _CACHE.setdefault(key, [])
            # keep at most 2 queued
            if len(_CACHE[key]) < 2:
                _CACHE[key].append(q)
    except Exception:
        # swallow prefetch errors
        pass

def _pop_or_fetch(key: Tuple[str, str, int], smap) -> dict:
    with _LOCK:
        qlist = _CACHE.get(key) or []
        if qlist:
            return qlist.pop(0)
    # fetch now if cache miss
    skill_id, difficulty, num_options = key
    skill = smap["skills"].get(skill_id) or next(iter(smap["skills"].values()))
    return LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=True)


def _top_category_id(skill_id: str, smap) -> str:
    skills = smap["skills"]
    cur = skills.get(skill_id)
    if not cur:
        return skill_id
    seen = set()
    while cur and cur.get("parent"):
        pid = cur.get("parent")
        if pid in seen:
            break
        seen.add(pid)
        cur = skills.get(pid)
    return cur.get("id") if cur else skill_id


# ----- Adaptivity helpers -----
from datetime import datetime, timezone
import math


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _decayed_mastery(state: dict) -> float:
    m = float(state.get("mastery", 0.0) or 0.0)
    last = state.get("last_seen_at")
    if not last or m <= 0:
        return m
    try:
        ts = last[:-1] if last.endswith("Z") else last
        t0 = datetime.fromisoformat(ts)
        dt_days = (datetime.now(timezone.utc) - t0.replace(tzinfo=timezone.utc)).total_seconds() / 86400.0
        k = math.log(2.0) / 7.0  # 7-day half-life
        return max(0.0, min(1.0, m * math.exp(-k * max(0.0, dt_days))))
    except Exception:
        return m


def _review_interval_days(mastery: float) -> float:
    # Coarse schedule: lower mastery -> shorter interval
    if mastery < 0.4:
        return 1.0
    if mastery < 0.6:
        return 3.0
    if mastery < 0.8:
        return 7.0
    return 14.0


def _next_skill_choice(smap, stats: dict, current_skill_id: str | None, last_correct: bool | None) -> tuple[str, str]:
    skills = smap["skills"]
    reason = "continue"
    # Utilities
    def _days_since_last(st: dict) -> float:
        last = st.get("last_seen_at")
        if not last:
            return float("inf")
        try:
            ts = last[:-1] if last.endswith("Z") else last
            t0 = datetime.fromisoformat(ts)
            return (datetime.now(timezone.utc) - t0.replace(tzinfo=timezone.utc)).total_seconds() / 86400.0
        except Exception:
            return float("inf")

    def _is_due(st: dict) -> bool:
        dm = _decayed_mastery(st)
        return _days_since_last(st) >= _review_interval_days(dm)

    # 1) Remediation if last answer wrong (single immediate follow-up)
    if (last_correct is False) and current_skill_id:
        return current_skill_id, "remediation"
    # Prepare per-skill states
    per_skill = (stats or {}).get("per_skill", {})
    # 2) Continue current only when due and not mastered
    if current_skill_id:
        st = per_skill.get(current_skill_id, {})
        dm = _decayed_mastery(st)
        if dm < 0.8 and _is_due(st):
            return current_skill_id, "continue-current"
    # 3) Review due by decay and interval
    due_candidates = []
    for sid, st in per_skill.items():
        if _is_due(st):
            days = _days_since_last(st)
            dm = _decayed_mastery(st)
            due_candidates.append((sid, dm, days))
    if due_candidates:
        due_candidates.sort(key=lambda x: (x[1], -x[2]))  # lowest mastery, longest wait first
        return due_candidates[0][0], "review-due"
    # 4) Progress to next new skill with prereqs satisfied
    # naive order: by same parent as current, else any
    def prereqs_satisfied(sid: str) -> bool:
        prereqs = skills.get(sid, {}).get("prereqs", []) or []
        for pid in prereqs:
            st = per_skill.get(pid, {})
            if _decayed_mastery(st) < 0.8:
                return False
        return True

    # prefer siblings of current
    cand = list(skills.keys())
    if current_skill_id:
        parent = skills.get(current_skill_id, {}).get("parent")
        cand = [s for s in cand if skills.get(s, {}).get("parent") == parent and s != current_skill_id] + \
               [s for s in cand if skills.get(s, {}).get("parent") != parent]
    for sid in cand:
        if not prereqs_satisfied(sid):
            continue
        st = per_skill.get(sid)
        if not st or _decayed_mastery(st) < 0.8:
            return sid, "advance-new"
    # fallback
    chosen = current_skill_id or next(iter(skills.keys()))
    return chosen, "fallback"


# Serve static UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
STATIC_DIR = os.path.abspath(STATIC_DIR)
if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/skills")
def api_skills() -> Dict[str, Any]:
    smap = load_skill_map()
    skills = [
        {"id": s["id"], "name": s["name"], "bloom": s.get("bloom", "")}
        for s in smap["skills"].values()
    ]
    # Sort by name for usability
    skills.sort(key=lambda x: x["name"].lower())
    return {"skills": skills}


@app.post("/api/user/upsert")
def api_user_upsert(payload: Dict[str, Any]) -> Dict[str, Any]:
    username = (payload.get("username") or "").strip()
    if not username:
        raise HTTPException(400, detail="username is required")
    try:
        return STORE.upsert_user(username)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@app.get("/api/user/by-name/{username}")
def api_user_by_name(username: str) -> Dict[str, Any]:
    prof = STORE.get_by_name(username)
    if not prof:
        raise HTTPException(404, detail="not found")
    return prof


@app.post("/api/generate")
def api_generate(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, Any]:
    smap = load_skill_map()
    skill_id = payload.get("skill_id") or "cog-learning-theories"
    qtype = (payload.get("type") or "mcq").lower()
    difficulty = (payload.get("difficulty") or "medium").lower()
    num_options = int(payload.get("num_options") or 5)
    rich = bool(payload.get("rich") or False)
    verify = bool(payload.get("verify") or False or os.getenv("TUTOR_VERIFY") == "1")
    use_templates = bool(payload.get("use_templates") or False or os.getenv("TUTOR_USE_TEMPLATES") == "1")
    user_id = payload.get("user_id")
    if qtype == "mcq":
        key = (skill_id, difficulty, num_options)
        if rich:
            # bypass minimal cache, fetch rich immediately
            skill = smap["skills"].get(skill_id) or next(iter(smap["skills"].values()))
            template = None
            if use_templates:
                templ_all = templates_for_skill(skill_id)
                if templ_all:
                    if user_id:
                        us = STORE.get(user_id)
                        tcounts = ((us.get("per_skill", {}).get(skill_id) or {}).get("template_counts") or {})
                        templ_all = sorted(templ_all, key=lambda t: int(tcounts.get(t["id"], 0)))
                    template = templ_all[0]
            q = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=False, template=template)
            if template:
                q.setdefault("template_id", template.get("id"))
        else:
            q = _pop_or_fetch(key, smap)
        # Rule-based validation; one retry on failure
        ok, errs = _validate_mcq(q, expected_len=num_options)
        if not ok:
            try:
                skill = smap["skills"].get(skill_id) or next(iter(smap["skills"].values()))
                q2 = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=not rich, template=(template if rich else None))
                ok2, _ = _validate_mcq(q2, expected_len=num_options)
                if ok2:
                    q = q2
                else:
                    # keep original but mark invalid reasons
                    q = q
            except Exception:
                pass
        # Two-stage self verification
        if verify and rich:
            try:
                ans = LLM.answer_mcq(q.get("stem", ""), q.get("options", []) or [])
                chosen = ans.get("chosen_index")
                if not (isinstance(chosen, int) and chosen == q.get("correct_index")):
                    q2 = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=False, template=(template if rich else None))
                    if _validate_mcq(q2, expected_len=num_options)[0]:
                        q = q2
            except Exception:
                pass
        # Optional anonymization (env or payload)
        anonymize = bool(payload.get("anonymize") or os.getenv("TUTOR_ANONYMIZE") == "1")
        if anonymize:
            try:
                uid = payload.get("user_id")
                q = _anonymize_question_for_user(q, uid, smap)
            except Exception:
                pass
        # Ensure item_id present
        if "item_id" not in q:
            q["item_id"] = _item_id_for_mcq(skill_id, q)
        q.setdefault("prompt_id", "mcq-rich-v1" if rich else "mcq-minimal-v1")
        # prefetch next in background
        if not rich:
            background_tasks.add_task(_prefetch, key, smap)
        return {"type": "mcq", "skill_id": skill_id, "question": q}
    elif qtype == "saq":
        skill = smap["skills"].get(skill_id) or next(iter(smap["skills"].values()))
        return {"type": "saq", "skill_id": skill_id, "question": LLM.generate_saq(skill, difficulty=difficulty)}
    else:
        raise HTTPException(400, detail="type must be 'mcq' or 'saq'")


@app.post("/api/record")
def api_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    smap = load_skill_map()
    user_id = payload.get("user_id")
    skill_id = payload.get("skill_id")
    correct = bool(payload.get("correct"))
    misconception_tag = payload.get("misconception_tag")
    item_id = payload.get("item_id")
    confidence = payload.get("confidence")
    time_ms = payload.get("time_to_answer_ms")
    template_id = payload.get("template_id")
    # Optional evidence/citations for hardening
    citations = payload.get("citations") or []
    citations_text = payload.get("citations_text") or ""
    stem = payload.get("stem")
    options = payload.get("options") or []
    correct_index = payload.get("correct_index")
    if not user_id or not skill_id:
        raise HTTPException(400, detail="user_id and skill_id are required")
    if skill_id not in smap["skills"]:
        raise HTTPException(404, detail=f"Unknown skill_id: {skill_id}")
    category_id = _top_category_id(skill_id, smap)
    # ----- Coverage + witness scoring (optional) -----
    def _tok(text: str) -> set[str]:
        return set([t for t in _re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if len(t) >= 3])

    credited = correct
    witness_pass = None
    coverage = None
    if os.getenv("TUTOR_REQUIRE_CITATIONS") == "1":
        try:
            tau = float(os.getenv("TUTOR_COVERAGE_TAU") or 0.4)
        except Exception:
            tau = 0.4
        # Gold text = correct option (+ optional rationale if present in payload)
        gold = ""
        try:
            if isinstance(options, list) and isinstance(correct_index, int) and 0 <= correct_index < len(options):
                gold = (options[correct_index] or "")
            if isinstance(payload.get("rationales"), list) and 0 <= int(correct_index) < len(payload["rationales"]):
                gold += "\n" + (payload["rationales"][int(correct_index)] or "")
        except Exception:
            pass
        gold_t = _tok(gold)
        cite_t = _tok(citations_text)
        coverage = (len(gold_t & cite_t) / max(1, len(gold_t))) if gold_t else 0.0
        # Witness: pick using only citations_text
        witness_idx = None
        if isinstance(options, list) and options:
            scores = []
            for i, opt in enumerate(options):
                ot = _tok(opt)
                score = len(ot & cite_t)
                scores.append((score, i))
            scores.sort(reverse=True)
            witness_idx = scores[0][1]
        witness_pass = (witness_idx == correct_index)
        credited = bool(correct and (coverage is not None and coverage >= tau) and (witness_pass is True))
    # Record credited result
    stats = STORE.record(user_id, skill_id, category_id, credited, misconception_tag=misconception_tag, template_id=template_id)
    # item-level aggregation
    if item_id:
        try:
            STORE.record_item_event(item_id=item_id, skill_id=skill_id, correct=correct, confidence=confidence, time_ms=time_ms)
        except Exception:
            pass
    out = {"ok": True, "stats": stats, "category_id": category_id}
    if os.getenv("TUTOR_REQUIRE_CITATIONS") == "1":
        out.update({"credited": credited, "coverage": coverage, "witness_pass": witness_pass})
    return out


@app.get("/api/user/{user_id}/stats")
def api_user_stats(user_id: str) -> Dict[str, Any]:
    return STORE.get(user_id)


@app.post("/api/grade/saq")
def api_grade_saq(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        stem = payload["stem"]
        expected_points = payload["expected_points"]
        model_answer = payload["model_answer"]
        student_answer = payload["student_answer"]
    except KeyError as e:
        raise HTTPException(400, detail=f"Missing field: {e}")
    llm = OpenAILLM()
    grading = llm.grade_saq(stem, expected_points, model_answer, student_answer)
    return grading


# ----- Item validation and fingerprint -----
import hashlib
import re as _re


def _item_id_for_mcq(skill_id: str, q: dict) -> str:
    stem = (q.get("stem") or "").strip()
    opts = q.get("options") or []
    ci = q.get("correct_index")
    h = hashlib.sha256()
    h.update(skill_id.encode("utf-8"))
    h.update(b"\n")
    h.update(stem.encode("utf-8"))
    h.update(b"\n")
    for o in opts:
        h.update((o or "").strip().encode("utf-8"))
        h.update(b"\n")
    h.update(str(ci).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def _validate_mcq(q: dict, expected_len: int | None = None) -> tuple[bool, list[str]]:
    errs: list[str] = []
    stem = (q.get("stem") or "").strip()
    opts = q.get("options") or []
    ci = q.get("correct_index")
    if not stem or len(stem.split()) < 3:
        errs.append("stem_too_short")
    if not isinstance(opts, list) or len(opts) < 2:
        errs.append("too_few_options")
    if expected_len is not None and len(opts) != expected_len:
        errs.append("option_len_mismatch")
    # duplicate options
    norm = [(_o or "").strip().lower() for _o in opts]
    if len(set(norm)) != len(norm):
        errs.append("duplicate_options")
    banned = {"all of the above", "none of the above", "all of these", "none of these"}
    if any(n in banned for n in norm):
        errs.append("banned_choice")
    if not isinstance(ci, int) or ci < 0 or ci >= len(opts):
        errs.append("bad_correct_index")
    # overly imbalanced lengths
    if opts:
        lens = [len((o or "").strip()) for o in opts]
        if max(lens) > 3 * (sum(lens) / len(lens) + 1):
            errs.append("imbalanced_lengths")
    return (len(errs) == 0, errs)

def _self_verify_mcq(q: dict) -> bool:
    try:
        ans = LLM.answer_mcq(q.get("stem", ""), q.get("options", []) or [])
        chosen = ans.get("chosen_index")
        return isinstance(chosen, int) and chosen == q.get("correct_index")
    except Exception:
        return True

@app.post("/api/next")
def api_next(payload: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, Any]:
    smap = load_skill_map()
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(400, detail="user_id required")
    current_skill_id = payload.get("current_skill_id")
    last_correct = payload.get("last_correct")
    difficulty = (payload.get("difficulty") or "medium").lower()
    num_options = int(payload.get("num_options") or 5)
    rich = bool(payload.get("rich") or False)
    verify = bool(payload.get("verify") or False or os.getenv("TUTOR_VERIFY") == "1")
    use_templates = bool(payload.get("use_templates") or False or os.getenv("TUTOR_USE_TEMPLATES") == "1")

    stats = STORE.get(user_id)
    next_skill_id, reason = _next_skill_choice(smap, stats, current_skill_id, last_correct)

    if (payload.get("type") or "mcq").lower() != "mcq":
        raise HTTPException(400, detail="only mcq supported for /api/next in this build")

    if rich:
        skill = smap["skills"].get(next_skill_id) or next(iter(smap["skills"].values()))
        template = None
        if use_templates and user_id:
            templ_all = templates_for_skill(next_skill_id)
            if templ_all:
                us = STORE.get(user_id)
                tcounts = ((us.get("per_skill", {}).get(next_skill_id) or {}).get("template_counts") or {})
                templ_all = sorted(templ_all, key=lambda t: int(tcounts.get(t["id"], 0)))
                template = templ_all[0]
        q = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=False, template=template)
        if template:
            q.setdefault("template_id", template.get("id"))
    else:
        key = (next_skill_id, difficulty, num_options)
        q = _pop_or_fetch(key, smap)
        background_tasks.add_task(_prefetch, key, smap)
    # validate + item_id
    ok, _ = _validate_mcq(q, expected_len=num_options)
    if not ok:
        try:
            skill = smap["skills"].get(next_skill_id) or next(iter(smap["skills"].values()))
            q2 = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=not rich, template=(template if rich else None))
            if _validate_mcq(q2, expected_len=num_options)[0]:
                q = q2
        except Exception:
            pass
    if verify and rich:
        try:
            if not _self_verify_mcq(q):
                q2 = LLM.generate_mcq(skill, difficulty=difficulty, num_options=num_options, minimal=False, template=(template if rich else None))
                if _validate_mcq(q2, expected_len=num_options)[0]:
                    q = q2
        except Exception:
            pass
    # Optional anonymization (env or payload)
    anonymize = bool(payload.get("anonymize") or os.getenv("TUTOR_ANONYMIZE") == "1")
    if anonymize:
        try:
            uid = payload.get("user_id")
            q = _anonymize_question_for_user(q, uid, smap)
        except Exception:
            pass
    if "item_id" not in q:
        q["item_id"] = _item_id_for_mcq(next_skill_id, q)
    q.setdefault("prompt_id", "mcq-rich-v1" if rich else "mcq-minimal-v1")
    return {"type": "mcq", "skill_id": next_skill_id, "reason": reason, "question": q}


@app.post("/api/item/flag")
def api_item_flag(payload: Dict[str, Any]) -> Dict[str, Any]:
    item_id = (payload.get("item_id") or "").strip()
    if not item_id:
        raise HTTPException(400, detail="item_id required")
    try:
        info = STORE.flag_item(item_id)
        return {"ok": True, "item": {"item_id": item_id, "flags": info.get("flags", 0)}}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


def _load_oer_map():
    try:
        import yaml  # type: ignore
    except Exception:
        return {"skills": {}}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "rt-psych-tutor", "oer_links.yaml")
    path = os.path.abspath(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {"skills": {}}
    except Exception:
        return {"skills": {}}


@app.get("/api/skill/{skill_id}/oer")
def api_skill_oer(skill_id: str) -> Dict[str, Any]:
    data = _load_oer_map()
    info = (data.get("skills", {}) or {}).get(skill_id)
    if not info:
        raise HTTPException(404, detail="not found")
    return {"skill_id": skill_id, **info}


@app.get("/api/dashboard")
def api_dashboard() -> Dict[str, Any]:
    out: Dict[str, Any] = {"top_missed_skills": [], "flagged_items": []}
    stats = STORE.data.get("stats", {})
    skill_agg: Dict[str, Dict[str, int]] = {}
    for u in stats.values():
        for sid, s in (u.get("per_skill", {}) or {}).items():
            a = skill_agg.setdefault(sid, {"correct": 0, "wrong": 0, "total": 0})
            a["correct"] += int(s.get("correct", 0))
            a["wrong"] += int(s.get("wrong", 0))
            a["total"] += int(s.get("total", 0))
    top = sorted(skill_agg.items(), key=lambda kv: (kv[1]["wrong"], -kv[1]["correct"]), reverse=True)
    out["top_missed_skills"] = [{"skill_id": sid, **vals} for sid, vals in top[:10]]
    items = STORE.data.get("items", {})
    flagged = sorted(items.items(), key=lambda kv: int((kv[1] or {}).get("flags", 0)), reverse=True)
    out["flagged_items"] = [{"item_id": iid, "flags": v.get("flags", 0), "p_value": v.get("p_value") } for iid, v in flagged[:10]]
    return out
