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


# More animated delivery and sentence-led cartoon visuals, Funny Kitties only.
replace(
    "src/generate_script.py",
    "Write a high-retention voiceover about this exact topic.",
    """Write a high-retention voiceover about this exact topic.
DELIVERY: Sound like an energetic, funny Shorts host. Use punchy spoken
sentences, lively reactions, playful emphasis, and varied rhythm. Avoid
flat documentary narration, slow openings, and long clauses. Make jokes
land clearly, but do not shout or use fake excitement.""",
    1,
)

replace(
    "src/main.py",
    "rate = f\"{random.randint(-6, -2)}%\"",
    "rate = f\"{random.randint(8, 14)}%\"",
    1,
)
replace(
    "src/main.py",
    """                orientation=(
                    "portrait"
                    if is_short
                    else "landscape"
                ),
            )""",
    """                orientation=(
                    "portrait"
                    if is_short
                    else "landscape"
                ),
                script=script_text,
            )""",
    1,
)

replace(
    "src/fetch_stock.py",
    "def _build_search_queries(topic: str) -> list:",
    "def _build_search_queries(topic: str, script: str | None = None) -> list:",
    1,
)
replace(
    "src/fetch_stock.py",
    "    queries = []\n\n    # ---------------------------------------------------------\n    # SPECIAL CASE: SEA CUCUMBER",
    """    queries = []
    cat_topic = bool(set(words).intersection({"cat", "cats", "kitten", "kittens", "feline", "felines"}))

    # For Funny Kitties, search cartoon animation using each narration
    # sentence as context, not just the broad topic.
    if cat_topic:
        queries.extend([
            "2D cartoon cat animation",
            "animated cartoon cats funny",
            "cartoon kitten animation",
        ])
        for sentence in re.split(r"(?<=[.!?])\\s+", script or ""):
            sentence_words = [
                word for word in _topic_words(sentence)
                if word not in {"cat", "cats", "kitten", "kittens", "feline", "felines"}
            ]
            if sentence_words:
                detail = " ".join(sentence_words[:4])
                queries.append(f"2D cartoon cat {detail}")
                queries.append(f"animated cartoon cats {detail}")

    # ---------------------------------------------------------
    # SPECIAL CASE: SEA CUCUMBER""",
    1,
)
replace(
    "src/fetch_stock.py",
    "    specific_animal = (\n        _is_specific_animal_topic(topic)\n    )",
    """    specific_animal = _is_specific_animal_topic(topic)
    cat_topic = bool(set(_topic_words(topic)).intersection(
        {"cat", "cats", "kitten", "kittens", "feline", "felines"}
    ))""",
    1,
)
replace(
    "src/fetch_stock.py",
    "        elif specific_animal:\n\n            fallback_queries = [",
    """        elif cat_topic:
            fallback_queries = [
                "2D cartoon cats animation",
                "funny animated cartoon kittens",
                "cartoon cat comedy animation",
            ]

        elif specific_animal:

            fallback_queries = [""",
    1,
)
replace(
    "src/fetch_stock.py",
    "def fetch_clips_for_topic(\n    topic: str,\n    out_dir: str,\n    count: int = 5,\n    orientation: str = \"landscape\",\n) -> list:",
    "def fetch_clips_for_topic(\n    topic: str,\n    out_dir: str,\n    count: int = 5,\n    orientation: str = \"landscape\",\n    script: str | None = None,\n) -> list:",
    1,
)
replace(
    "src/fetch_stock.py",
    "    queries = _build_search_queries(\n        topic\n    )",
    "    queries = _build_search_queries(topic, script=script)",
    1,
)

print("Funny Kitties: energetic narration and sentence-led cartoon search enabled.")

print("Funny Kitties Groq preparation completed.")
