from pathlib import Path
from types import SimpleNamespace
import os
import requests

ROOT = Path(__file__).resolve().parents[1]

def replace(path, old, new, expected=1):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    count = s.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} match(es), found {count}: {old[:100]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"Patched {path}")

# Replace Gemini's direct client with a tiny Groq-compatible adapter.
replace("src/generate_script.py",
    "try:\n    from google import genai as genai_client\n    GEMINI_AVAILABLE = True\nexcept ImportError:\n    GEMINI_AVAILABLE = False",
    "from src.groq_compat import GroqClient\nGEMINI_AVAILABLE = True")
replace("src/generate_script.py", "genai_client.Client(", "GroqClient(", 5)
replace("src/generate_script.py", 'os.environ.get("GEMINI_API_KEY")', 'os.environ.get("GROQ_API_KEY")', 5)
replace("src/generate_script.py", 'os.environ["GEMINI_API_KEY"]', 'os.environ["GROQ_API_KEY"]', 5)
replace("src/generate_script.py", 'GEMINI_MODEL = "gemini-3.5-flash-lite"', 'GEMINI_MODEL = "openai/gpt-oss-120b"')

# Rename Gemini-facing identifiers and messages in the Funny Kitties runtime.
p = ROOT / "src/generate_script.py"
s = p.read_text(encoding="utf-8")
s = s.replace("GEMINI_AVAILABLE", "GROQ_AVAILABLE")
s = s.replace("GEMINI_MODEL", "GROQ_MODEL")
s = s.replace("generate_with_gemini", "generate_with_groq")
s = s.replace("Gemini", "Groq").replace("gemini", "groq")
s = s.replace("from google.genai import types", "from types import SimpleNamespace")
s = s.replace("types.GenerateContentConfig(", "SimpleNamespace(")
p.write_text(s, encoding="utf-8")
print("Renamed Funny Kitties generator provider references to Groq.")

# Dynamic channel settings replace the hard-coded wildlife/science prompts.

# Remove the generic wildlife-documentary fallback.
p = ROOT / "src/generate_script.py"
s = p.read_text(encoding="utf-8")
start = s.index("def generate_with_template(")
end = s.index("\n\ndef generate_script(", start)
fallback = '''def generate_with_template(topic: str, config: dict, length_seconds: int, language: str) -> dict:
    script = (
        "These cats woke up and chose chaos. First, one tiny troublemaker "
        "turns a cardboard box into a luxury apartment. Then another cat "
        "acts personally betrayed by a closed door. And just when you think "
        "the drama is over, a third cat starts a completely unnecessary "
        "parkour championship. Honestly, the humans are just guests here."
    )
    return {"title": topic.strip(), "script": script}
'''
s = s[:start] + fallback + s[end:]
p.write_text(s, encoding="utf-8")

# Groq adapter for the existing chat-shaped generator interface.
(ROOT / "src/groq_compat.py").write_text(r"""
import os
from types import SimpleNamespace
import requests

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

class _Chat:
    def __init__(self, model=None, config=None):
        self.model = model or MODEL
        self.config = config

    def send_message(self, prompt):
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        limit = getattr(self.config, "max_output_tokens", None) or 3000
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": min(int(limit), 12000),
                "temperature": 0.85,
            },
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"Groq API {response.status_code}: {response.text[:500]}")
        content = response.json()["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("Groq returned an empty response")
        return SimpleNamespace(text=content, candidates=[])

class _Chats:
    def create(self, model=None, config=None):
        return _Chat(model, config)

class GroqClient:
    def __init__(self, api_key=None):
        self.chats = _Chats()
""".lstrip(), encoding="utf-8")

# Use only cat-only discovery; seeded topics and Groq topic generation are the sources.
replace("src/main.py", "clip_count = 15 if is_short else 30", "clip_count = 5 if is_short else 30")

print("Funny Kitties Groq preparation completed.")
