from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def replace(path, old, new, expected=1):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n != expected:
        raise RuntimeError(f"{path}: expected {expected} matches, found {n}: {old[:90]!r}")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print(f"Funny Kitties patch applied: {path}")

# The engine is checked out fresh for every run. These runtime-only patches
# keep Cosmic Curious untouched while routing this channel through Groq.
replace(
    "src/generate_script.py",
    "try:\n    from google import genai as genai_client\n    GEMINI_AVAILABLE = True\nexcept ImportError:\n    GEMINI_AVAILABLE = False",
    "from src.groq_compat import GroqClient\nGEMINI_AVAILABLE = True",
)
replace(
    "src/generate_script.py",
    "genai_client.Client(",
    "GroqClient(",
    expected=4,
)
replace(
    "src/generate_script.py",
    'os.environ.get("GEMINI_API_KEY")',
    'os.environ.get("GROQ_API_KEY")',
    expected=4,
)
replace(
    "src/generate_script.py",
    'os.environ["GEMINI_API_KEY"]',
    'os.environ["GROQ_API_KEY"]',
    expected=4,
)
replace(
    "src/generate_script.py",
    'GEMINI_MODEL = "gemini-3.5-flash-lite"',
    'GEMINI_MODEL = "openai/gpt-oss-120b"',
)
replace(
    "src/generate_script.py",
    'NICHE: Animals & wildlife facts',
    "NICHE: {config.get('niche', 'Funny cat compilations')}",
)
replace(
    "src/generate_script.py",
    "Every topic must concern real animals or wildlife. Explore different\nspecies, senses, behaviors, survival adaptations, intelligence, predators,\nprey, rare animals, deep-sea animals, and unusual animal relationships.\n\nDo not generate space, astronomy, NASA, unrelated science, or fictional\nanimal abilities. Avoid duplicate subjects and close rewordings.",
    "{config.get('topic_generation_rules', 'Generate only funny cat compilation topics. Avoid other animals and duplicates.')}",
)
replace(
    "src/generate_script.py",
    "NICHE: Animals and wildlife facts.\nTONE: Curious, surprising, energetic, punchy, conversational.",
    "NICHE: {config.get('niche', 'Funny cat compilations')}\nTONE: {config.get('tone', 'funny, playful, energetic and conversational')}",
)
replace(
    "src/generate_script.py",
    "CONTENT RULES:\n- The entire script must be about animals or wildlife.\n- Use real, well-established facts.\n- Never invent statistics, discoveries, quotes, experiments, behaviors,\n  or abilities.\n- Never exaggerate abilities or present speculation as fact.\n- Do not write about space, planets, stars, galaxies, black holes,\n  astronomy, NASA, the universe, or unrelated physics.\n- Stay focused on the specific animal and topic.\n- Do not invent a \"shocking\" fact just to make the script exciting.",
    "CONTENT RULES:\n{config.get('script_rules', 'Write funny English narration for a compilation of several different cats. Describe harmless, relatable cat behavior. Never invent a specific event or claim that generic stock footage shows a particular event.')}",
)
replace(
    "src/generate_script.py",
    "- Start with a concrete, surprising fact or consequence about the animal.",
    "- Open with a funny line that introduces a compilation of 4-5 different cats.",
)
replace(
    "src/generate_script.py",
    "- Every sentence must introduce a new fact, escalation, contrast, or payoff.",
    "- Each section must introduce a different cat or funny situation, with a fresh joke or punchline.",
)
replace(
    "src/generate_script.py",
    "- Build one clear curiosity loop and resolve it before the ending.",
    "- Structure the narration as 4-5 connected mini-scenes with smooth spoken transitions.",
)
replace(
    "src/generate_script.py",
    "- The payoff must be specific and supported by a real fact.",
    "- Every mini-scene must have a clear comedic payoff based on familiar cat behavior.",
)
replace(
    "src/generate_script.py",
    'cta = "Subscribe for more amazing animal facts every day! 🐾"',
    'cta = "Subscribe for more funny cat compilations! 🐾"',
)
replace(
    "src/generate_script.py",
    "What makes this topic worth investigating is the animal behavior behind it: {topic}. The important question is what the animal actually does, and why. Real wildlife can be surprising, but the explanation matters just as much as the surprise. The most interesting detail is the one that the evidence can support, not a claim that merely sounds impressive.",
    "This cat woke up and immediately chose chaos. And it is not alone. From dramatic reactions to completely unnecessary parkour, these cats are making ordinary life look like a comedy show. Just wait until the last one.",
)
replace(
    "src/generate_script.py",
    "if GEMINI_AVAILABLE and os.environ.get(\"GROQ_API_KEY\"):",
    "if os.environ.get(\"GROQ_API_KEY\"):",
)

# A small compatibility client lets the existing generator use Groq's
# OpenAI-compatible Chat Completions endpoint without changing Cosmic code.
(ROOT / "src/groq_compat.py").write_text(r'''
import os
from types import SimpleNamespace
import requests

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

class _Chat:
    def __init__(self, model, config=None):
        self.model = model or GROQ_MODEL
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
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("Groq returned an empty response")
        return SimpleNamespace(text=content, candidates=[])

class _Chats:
    def create(self, model=None, config=None):
        return _Chat(model, config)

class GroqClient:
    def __init__(self, api_key=None):
        self.chats = _Chats()
'''.lstrip(), encoding="utf-8")

# Central agent gateway: all editorial agents use Groq on this channel.
(ROOT / "src/gemini_client.py").write_text(r'''
import os
import requests

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

def generate(prompt: str, max_output_tokens: int | None = None, model: str = GROQ_MODEL) -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model or GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": min(int(max_output_tokens or 3000), 12000),
            "temperature": 0.75,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Groq API {response.status_code}: {response.text[:500]}")
    content = response.json()["choices"][0]["message"]["content"]
    if not content:
        raise RuntimeError("Groq returned an empty response")
    return content.strip()

def generate_response(prompt: str, max_output_tokens: int | None = None, model: str = GROQ_MODEL):
    return type("GroqResponse", (), {"text": generate(prompt, max_output_tokens, model)})()
'''.lstrip(), encoding="utf-8")

# Remove science-documentary instructions from the shared editorial agent prompts.
replace("src/agents/retention_agent.py",
    '"not an AI narrator reading facts, not a documentary narrator or an AI reading a Wikipedia summary, without weakening scientific accuracy."',
    '"with funny, natural, playful cat-compilation narration, not a wildlife documentary."',
    expected=0) if False else None

replace("src/agents/retention_agent.py",
    '"Improve pacing AND make the script sound like a real person "\n        "talking, not an AI narrator reading facts, without weakening "\n        "scientific accuracy."',
    '"Improve pacing and make the narration sound like a funny human host "\n        "commenting on a compilation of different cats."')
replace("src/agents/retention_agent.py",
    "Rewrite it to sound like a genuine person explaining this to a friend,\nnot a documentary narrator or an AI reading a Wikipedia summary. Concretely:",
    "Rewrite it as a funny, natural voiceover for a compilation of several different cats. Keep each scene distinct, use smooth transitions, and make the jokes sound conversational. Concretely:")
replace("src/agents/retention_agent.py",
    "Keep every fact and number exactly as accurate as the original -- this\nis a tone/phrasing/structure rewrite, not a content rewrite.",
    "Do not invent specific events, quotes, or actions that cannot be inferred from generic cat footage. Preserve the sequence and overall meaning of the scenes.")
replace("src/agents/hook_agent.py",
    '"Create immediate, accurate, non-clickbait openings for science Shorts."',
    '"Create funny, instantly engaging openings for funny cat compilation Shorts."')
replace("src/agents/hook_agent.py",
    "REQUIRED PRINCIPLE (backed by real view/retention data from this exact\nchannel -- not a style preference): the opening sentence MUST state the\nsingle most concrete, jaw-dropping claim of the whole video immediately,\nwith ZERO scene-setting, teasing, or wind-up. Either direct \"you/your\"\nframing or a bold third-person claim works -- what matters is stating the\nactual payoff in sentence one, not building up to it.",
    "The first sentence should be a short, funny line that sets up the cat compilation immediately. Avoid questions, generic greetings, and slow introductions. Make it sound like a witty human host.")
replace("src/agents/hook_agent.py",
    '("clarity", "curiosity", "retention", "science_safety")',
    '("clarity", "curiosity", "retention")',
    expected=0) if False else None
replace("src/agents/seo_agent.py",
    '"Optimize title/description/tags for discovery without spam or scientific misrepresentation."',
    '"Optimize titles, tags and hashtags for funny cat compilation Shorts. Keep everything cat-specific, natural and non-spammy."')
replace("src/agents/thumbnail_agent.py",
    '"Design high-contrast, curiosity-driven thumbnails that accurately represent the video."',
    '"Design bold, funny thumbnails for cat compilation videos, with expressive cats and short readable meme-style text."')

# Cat-only trend research. Reddit and YouTube signals are constrained to cats.
(ROOT / "src/trend_scout.py").write_text(r'''
from src.gemini_client import generate
import os
import requests

def get_trending_topic(channel_id: str, config: dict):
    if not config.get("trend_aware"):
        return None
    signals = []
    try:
        r = requests.get("https://www.reddit.com/r/cats/top.json",
            params={"t":"week","limit":12},
            headers={"User-Agent":"FunnyKittiesTopicScout/1.0"}, timeout=12)
        if r.ok:
            signals = [x["data"]["title"] for x in r.json()["data"]["children"]
                       if not x["data"].get("stickied")]
    except Exception:
        pass
    if not signals:
        return None
    prompt = f"""You are a topic researcher for Funny Kitties, an English funny-cat Shorts channel.
Create ONE fresh, specific idea for a funny compilation of 4-5 different cats.
Use these cat-only trend signals for inspiration, but do not copy titles:
{chr(10).join('- '+s for s in signals)}
Rules: cats and kittens only; harmless comedy only; no dogs, wildlife, other animals,
science facts, animal documentaries, injuries, cruelty or dangerous stunts.
Return only one short topic line."""
    topic = generate(prompt, max_output_tokens=250).strip().strip('"')
    if not topic or "cat" not in topic.lower() and "kitten" not in topic.lower():
        return None
    return topic
'''.lstrip(), encoding="utf-8")

# Keep only 4-5 long, readable cat scenes instead of 15 rapid cuts.
replace("src/main.py", "clip_count = 15 if is_short else 30", "clip_count = 5 if is_short else 30")
replace("src/main.py",
    'cta = "Subscribe for more amazing animal facts every day! 🐾"',
    'cta = "Subscribe for more funny cat compilations! 🐾"')

print("Funny Kitties Groq and cat-comedy preparation completed.")
