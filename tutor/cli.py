from __future__ import annotations
import argparse
import json
import sys
from typing import Any

from tutor.skill_map import load_skill_map, skill_summary
from tutor.llm_openai import OpenAILLM
from tutor.utils import dumps, load_env_dotenv_fallback


def cmd_generate_mcq(args: argparse.Namespace) -> int:
    smap = load_skill_map(args.skill_map)
    skill = smap["skills"].get(args.skill_id)
    if not skill:
        print(f"Skill not found: {args.skill_id}")
        return 2
    llm = OpenAILLM()
    res = llm.generate_mcq(skill, difficulty=args.difficulty)
    print(dumps({"skill": skill_summary(skill), "mcq": res}))
    return 0


def cmd_generate_saq(args: argparse.Namespace) -> int:
    smap = load_skill_map(args.skill_map)
    skill = smap["skills"].get(args.skill_id)
    if not skill:
        print(f"Skill not found: {args.skill_id}")
        return 2
    llm = OpenAILLM()
    res = llm.generate_saq(skill, difficulty=args.difficulty)
    print(dumps({"skill": skill_summary(skill), "saq": res}))
    return 0


def cmd_grade_saq(args: argparse.Namespace) -> int:
    load_env_dotenv_fallback()
    llm = OpenAILLM()
    expected_points: list[dict[str, Any]] = json.loads(args.expected_points)
    res = llm.grade_saq(args.stem, expected_points, args.model_answer, args.student_answer)
    print(dumps(res))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    # Minimal demo: generate one MCQ for a common skill
    smap = load_skill_map(args.skill_map)
    skill = smap["skills"].get("cog-learning-theories") or next(iter(smap["skills"].values()))
    llm = OpenAILLM()
    mcq = llm.generate_mcq(skill, difficulty="medium")
    print("Generated MCQ for:", skill_summary(skill))
    print(dumps(mcq))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    # Confirm .env and key presence without printing it
    load_env_dotenv_fallback()
    import os
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not found in environment/.env")
        return 2
    masked = key[:6] + "..." + key[-4:]
    print("Found OPENAI_API_KEY:", masked)
    llm = OpenAILLM()
    info = llm.verify_key_and_model()
    print(dumps(info))
    ok = info.get("chat_ok") and info.get("models_list_ok")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]

    p = argparse.ArgumentParser(prog="tutor", description="Real-time psychology tutor CLI (OpenAI-backed)")
    p.add_argument("--skill-map", dest="skill_map", default="docs/rt-psych-tutor/skill_map.psych101.yaml")

    sp = p.add_subparsers(dest="cmd", required=True)

    gmcq = sp.add_parser("generate-mcq", help="Generate an MCQ for a skill")
    gmcq.add_argument("--skill-id", required=True)
    gmcq.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    gmcq.set_defaults(func=cmd_generate_mcq)

    gsaq = sp.add_parser("generate-saq", help="Generate an SAQ for a skill")
    gsaq.add_argument("--skill-id", required=True)
    gsaq.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    gsaq.set_defaults(func=cmd_generate_saq)

    grade = sp.add_parser("grade-saq", help="Grade a short answer against expected points")
    grade.add_argument("--stem", required=True)
    grade.add_argument("--expected-points", required=True, help="JSON list of {key,required}")
    grade.add_argument("--model-answer", required=True)
    grade.add_argument("--student-answer", required=True)
    grade.set_defaults(func=cmd_grade_saq)

    demo = sp.add_parser("demo", help="Quick demo: generate one MCQ")
    demo.set_defaults(func=cmd_demo)

    verify = sp.add_parser(
        "verify", help="Verify API key works and model is callable"
    )
    verify.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
