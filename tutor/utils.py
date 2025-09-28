import json, os, re


def load_env_dotenv_fallback(path: str = ".env") -> None:
    """Load .env into os.environ if keys not already set.
    Tries python-dotenv if available; otherwise parses simple KEY=VALUE lines.
    """
    # Try python-dotenv if installed
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path)
        return
    except Exception:
        pass

    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass


def extract_json(text: str):
    """Extract a JSON object from model output, tolerating code fences."""
    # Strip code fences
    fenced = re.search(r"```(?:json)?\n(.*?)\n```", text, re.S)
    if fenced:
        text = fenced.group(1)
    # Find first { .. matching brace count
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")
    # Attempt to parse progressively
    for end in range(len(text), start, -1):
        chunk = text[start:end].strip()
        try:
            return json.loads(chunk)
        except Exception:
            continue
    raise ValueError("Failed to parse JSON from text")


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

