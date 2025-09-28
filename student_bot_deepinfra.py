#!/usr/bin/env python3
"""
Synthetic Student — DeepInfra Harness

This script runs a "synthetic student" that:
  1) Upserts a user by username against the tutor server.
  2) Requests adaptive MCQs via /api/next.
  3) Asks an OpenAI-compatible LLM (DeepInfra or OpenAI) to pick A/B/C/D/E.
  4) Records correctness, confidence, and response time via /api/record.

Usage examples (ensure DEEPINFRA_API_KEY is in your environment):
  export $(grep -v '^#' .env | xargs)  # load env vars into current shell
  python student_bot_deepinfra.py --username bot01 --n 300
  python student_bot_deepinfra.py --username bot02 --n 500 --model meta-llama/Meta-Llama-3.1-8B-Instruct
  python student_bot_deepinfra.py --username bot03 --n 500 --skill cog-learning-theories

Notes:
- The tutor generation model remains hard-locked to OpenAI's gpt-5-nano-2025-08-07 inside the server.
- This harness supports DeepInfra's OpenAI-compatible endpoint (open-source models) and OpenAI provider.
- For "closed-book" simulation, pass --closed-book and --rich so the student only uses accumulated NOTES built from tutor feedback (starts empty).
"""

from __future__ import annotations
import argparse
import os
import random
import statistics
import time
from typing import Any, Dict, List, Tuple

import requests
from openai import OpenAI
import json
import re
from collections import Counter


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing required environment variable: {name}")
    return val


def upsert_user(tutor_url: str, username: str) -> str:
    r = requests.post(f"{tutor_url}/api/user/upsert", json={"username": username}, timeout=30)
    r.raise_for_status()
    return r.json()["user_id"]


def get_next_item(tutor_url: str, user_id: str, skill_id: str | None = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "user_id": user_id,
        "type": "mcq",
        "difficulty": "medium",
        "num_options": 5,
        "verify": True,
        "use_templates": True,
        "rich": True,
    }
    if skill_id:
        body["current_skill_id"] = skill_id
    r = requests.post(f"{tutor_url}/api/next", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def record_answer(
    tutor_url: str,
    user_id: str,
    skill_id: str,
    correct: bool,
    *,
    item_id: str | None = None,
    confidence: int | None = None,
    time_ms: int | None = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"user_id": user_id, "skill_id": skill_id, "correct": bool(correct)}
    if item_id is not None:
        body["item_id"] = item_id
    if isinstance(confidence, int):
        body["confidence"] = int(confidence)
    if isinstance(time_ms, int):
        body["time_to_answer_ms"] = int(time_ms)
    r = requests.post(f"{tutor_url}/api/record", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_with_llm(stem: str, options: List[str], *, model: str, client: OpenAI) -> int:
    letters = "ABCDE"[: len(options)]
    prompt = (
        "You are a struggling Intro Psych student. Read the question and pick ONE answer.\n"
        "Respond with a single character: A, B, C, D, or E. No explanation.\n\n"
        f"Question: {stem}\n"
        "Options:\n" + "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    for ch in text:
        if ch in letters:
            return letters.index(ch)
    # Fallback: random guess if no valid letter found
    return random.randrange(len(options))


def _extract_json(text: str) -> Dict[str, Any]:
    # Try code fence
    m = re.search(r"```(?:json)?\n(.*?)\n```", text, re.S)
    if m:
        text = m.group(1)
    # Find first JSON object
    i = text.find("{")
    if i == -1:
        raise ValueError("no json")
    for j in range(len(text), i, -1):
        chunk = text[i:j].strip()
        try:
            return json.loads(chunk)
        except Exception:
            pass
    raise ValueError("parse error")


def pick_with_notes(stem: str, options: List[str], notes: str, *, model: str, client: OpenAI) -> int:
    # Closed-book chooser: only use NOTES; return JSON
    prompt = (
        "You are a novice student with no prior knowledge beyond NOTES.\n"
        "Use ONLY the NOTES to decide; if NOTES are insufficient, guess randomly.\n"
        "Return only JSON with keys: chosen_index (int 0..N-1), used_notes (boolean). No explanation.\n\n"
        f"NOTES:\n{notes.strip() or '(none)'}\n\n"
        f"Question: {stem}\n"
        "Options (0-indexed):\n" + "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        data = _extract_json(text)
        ci = int(data.get("chosen_index", -1))
        if 0 <= ci < len(options):
            return ci
    except Exception:
        pass
    # Fallback to letter/guess path
    return pick_with_llm(stem, options, model=model, client=client)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or '').lower())


def pick_with_algo(notes: str, options: List[str]) -> int:
    # Simple overlap scorer between NOTES and each option
    ntoks = _tokenize(notes)
    if not ntoks:
        return random.randrange(len(options))
    bag = Counter(ntoks)
    best = 0
    best_score = -1.0
    for i, opt in enumerate(options):
        otoks = _tokenize(opt)
        score = sum(bag.get(t, 0) for t in otoks)
        if score > best_score:
            best_score = score
            best = i
    return best


def choose_citations(notes_lines: List[str], focus_tokens: List[str], k: int = 3) -> Tuple[List[Dict[str, Any]], str]:
    if not notes_lines or not focus_tokens:
        return [], ""
    fset = set(t for t in focus_tokens if len(t) >= 3)
    scored: List[Tuple[int, int]] = []  # (score, idx)
    for i, line in enumerate(notes_lines):
        lt = set(_tokenize(line))
        score = len(lt & fset)
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[:k]
    cites = []
    chunks = []
    for _, idx in picked:
        cites.append({"doc": "NOTES", "line": idx, "text": notes_lines[idx]})
        chunks.append(notes_lines[idx])
    return cites, "\n".join(chunks)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a synthetic student against the tutor (DeepInfra or OpenAI provider).")
    ap.add_argument("--username", required=True, help="Username for the synthetic student")
    ap.add_argument("--n", type=int, default=200, help="Number of items to answer")
    ap.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        help="Model id (OpenAI-compatible). Use provider-specific names.",
    )
    ap.add_argument(
        "--provider",
        choices=["deepinfra", "openai"],
        default="deepinfra",
        help="LLM API provider: DeepInfra (default) or OpenAI",
    )
    ap.add_argument("--skill", default=None, help="Optional fixed skill_id to focus on")
    ap.add_argument("--tutor-url", default="http://localhost:8000", help="Tutor base URL")
    ap.add_argument("--fast", action="store_true", help="Skip think-time sleeps for rapid simulations")
    ap.add_argument("--random-pick", action="store_true", help="Pick answers randomly instead of calling DeepInfra")
    ap.add_argument("--closed-book", action="store_true", help="Use only accumulated NOTES to answer; starts with no knowledge")
    ap.add_argument("--notes-file", default=None, help="Path to persist/load student NOTES (text)")
    ap.add_argument("--algo", action="store_true", help="Use algorithmic NOTES-overlap student (no LLM)")
    args = ap.parse_args(argv)

    # Set up OpenAI-compatible client unless using random picks
    client = None
    model_name = args.model
    if not args.random_pick and not args.algo:
        if args.provider == "deepinfra":
            api_key = require_env("DEEPINFRA_API_KEY")
            client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        else:
            api_key = require_env("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            if model_name.startswith("openai/"):
                model_name = model_name.split("/", 1)[1]

    # Load NOTES (for closed-book mode)
    notes_path = args.notes_file or (f"data/student_notes_{args.username}.txt")
    if args.closed_book:
        os.makedirs(os.path.dirname(notes_path), exist_ok=True)
        try:
            with open(notes_path, "r", encoding="utf-8") as f:
                notes_lines = [line.rstrip("\n") for line in f]
        except Exception:
            notes_lines = []
    else:
        notes_lines = []

    # Upsert or get user id
    user_id = upsert_user(args.tutor_url, args.username)

    corrects = 0
    latencies: List[int] = []

    last_skill = args.skill
    for _ in range(args.n):
        nxt = get_next_item(args.tutor_url, user_id, last_skill)
        skill_id = nxt.get("skill_id")
        q = nxt.get("question") or {}
        stem = q.get("stem") or ""
        options = q.get("options") or []
        key = q.get("correct_index")
        if not options or not isinstance(key, int):
            # Skip malformed item
            continue
        # Simulate thinking time (~1–20s lognormal-ish)
        if args.fast:
            think_ms = 1
        else:
            think_ms = int(max(600, random.lognormvariate(7.1, 0.35)))
            time.sleep(think_ms / 1000.0)
        if args.random_pick:
            pick = random.randrange(len(options))
        elif args.algo:
            notes_text = "\n".join(notes_lines[-200:])
            pick = pick_with_algo(notes_text, options)
        else:
            if args.closed_book:
                notes_text = "\n".join(notes_lines[-200:])
                pick = pick_with_notes(stem, options, notes_text, model=model_name, client=client)
            else:
                pick = pick_with_llm(stem, options, model=model_name, client=client)
        is_correct = (pick == key)
        conf = random.choices([2, 3, 4], weights=[2, 5, 3])[0]
        # Prepare citations and payload extras for scoring (coverage + witness)
        cites: List[Dict[str, Any]] = []
        cites_text = ""
        if args.closed_book:
            # Focus tokens from picked option + stem
            focus = _tokenize(stem) + _tokenize(options[pick] if 0 <= pick < len(options) else "")
            cites, cites_text = choose_citations(notes_lines[-200:], focus)
        body: Dict[str, Any] = {
            "user_id": user_id,
            "skill_id": skill_id,
            "correct": bool(is_correct),
            "item_id": q.get("item_id"),
            "confidence": int(conf),
            "time_to_answer_ms": int(think_ms),
            # Extras for server-side coverage/witness scoring
            "stem": stem,
            "options": options,
            "correct_index": key,
            "rationales": q.get("rationales"),
            "citations": cites,
            "citations_text": cites_text,
        }
        r = requests.post(f"{args.tutor_url}/api/record", json=body, timeout=30)
        r.raise_for_status()
        # Update NOTES using rich feedback when available
        if args.closed_book:
            try:
                ci = int(key)
                rationale = ""
                if isinstance(q.get("rationales"), list) and 0 <= ci < len(q["rationales"]):
                    rationale = q["rationales"][ci] or ""
                corr_opt = (options[ci] if 0 <= ci < len(options) else "")
                notes_lines.append(f"Q: {stem}")
                notes_lines.append(f"Correct: {corr_opt}")
                if rationale:
                    notes_lines.append(f"Why: {rationale}")
                notes_lines.append("")
                with open(notes_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(notes_lines[-1000:]))
            except Exception:
                pass
        # carry forward the current skill to aid adaptivity
        last_skill = skill_id
        corrects += int(is_correct)
        latencies.append(think_ms)

    acc = corrects / max(1, args.n)
    mean_ms = int(statistics.mean(latencies)) if latencies else 0
    print({"user": args.username, "n": args.n, "acc": round(acc, 3), "mean_ms": mean_ms})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
