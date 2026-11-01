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

# Dynamic channel settings replace the hard-coded wildlife/science prompts.
replace("src/generate_script.py", "NICHE: Animals & wildlife facts", "NICHE: {config.get('niche', 'Funny cat compilations')}")
replace("src/generate_script.py",
    "Every topic must concern real animals or wildlife. Explore different\nspecies, senses, behaviors, survival adaptations, intelligence, predators,\nprey, rare animals, deep-sea animals, and unusual animal relationships.\n\nDo not generate space, astronomy, NASA, unrelated science, or fictional\nanimal abilities. Avoid duplicate subjects and close rewordings.",
    "{config.get('topic_generation_rules', 'Generate only funny cat compilation topics. Avoid other animals and duplicates.')}")
replace("src/generate_script.py",
    "NICHE: Animals and wildlife facts.\nTONE: Curious, surprising, energetic, punchy, conversational.",
    "NICHE: {config.get('niche', 'Funny cat compilations')}\nTONE: {config.get('tone', 'funny, playful, energetic and conversational')}")
replace("src/generate_script.py",
    "CONTENT RULES:\n- The entire script must be about animals or wildlife.\n- Use real, well-established facts.\n- Never invent statistics, discoveries, quotes, experiments, behaviors,\n  or abilities.\n- Never exaggerate abilities or present speculation as fact.\n- Do not write about space, planets, stars, galaxies, black holes,\n  astronomy, NASA, the universe, or unrelated physics.\n- Stay focused on the specific animal and topic.\n- Do not invent a \"shocking\" fact just to make the script exciting.",
    "CONTENT RULES:\n{config.get('script_rules', 'Write funny English narration for a compilation of several different cats. Describe harmless, relatable cat behavior. Never invent a specific event or claim that generic stock footage shows a particular event.')}")
replace("src/generate_script.py",
    "- Start with a concrete, surprising fact or consequence about the animal.",
    "- Open with a funny line that introduces a compilation of 4-5 different cats.")
replace("src/generate_script.py",
    "- Every sentence must introduce a new fact, escalation, contrast, or payoff.",
    "- Each section must introduce a different cat or funny situation, with a fresh joke or punchline.")
replace("src/generate_script.py",
    "- Build one clear curiosity loop and resolve it before the ending.",
    "- Structure the narration as 4-5 connected mini-scenes with smooth spoken transitions.")
replace("src/generate_script.py",
    "- The payoff must be specific and supported by a real fact.",
    "- Every mini-scene must have a clear comedic payoff based on familiar cat behavior.")

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
