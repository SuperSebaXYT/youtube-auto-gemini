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


# Funny Kitties only: lively narration and action-matched cartoon cat footage.
replace(
    "src/generate_script.py",
    "Write a high-retention voiceover about this exact topic.",
    """Write a high-retention voiceover about this exact topic.
DELIVERY: Speak like an energetic, funny Shorts host. Use lively reactions,
punchy phrasing, playful emphasis and varied rhythm. Keep it natural, not
shouty or flat.""",
    1,
)

replace(
    "src/main.py",
    "rate = f\"{random.randint(-6, -2)}%\"",
    "rate = f\"+{random.randint(10, 16)}%\"",
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
    """    if cat_topic:
        queries.extend([
            "funny cats",
            "cats being funny",
            "cats playing",
            "funny kittens",
            "cat reactions",
            "cats at home",
        ])""",
    """    if cat_topic:
        # Match searches to actions described by the narration.
        action_terms = {
            "jump": "jumping", "jumps": "jumping", "jumped": "jumping",
            "leap": "leaping", "leaps": "leaping",
            "run": "running", "runs": "running", "ran": "running",
            "chase": "chasing", "chases": "chasing", "chased": "chasing",
            "sleep": "sleeping", "sleeps": "sleeping", "slept": "sleeping",
            "climb": "climbing", "climbs": "climbing", "climbed": "climbing",
            "fall": "falling", "falls": "falling", "fell": "falling",
            "play": "playing", "plays": "playing", "played": "playing",
            "hide": "hiding", "hides": "hiding", "hid": "hiding",
            "pounce": "pouncing", "pounces": "pouncing",
            "scratch": "scratching", "scratches": "scratching",
            "swat": "swatting", "swats": "swatting",
            "push": "pushing", "pushes": "pushing",
            "knock": "knocking", "knocks": "knocking",
            "steal": "stealing", "steals": "stealing",
            "eat": "eating", "eats": "eating",
            "drink": "drinking", "drinks": "drinking",
            "lick": "licking", "licks": "licking",
            "meow": "meowing", "meows": "meowing",
            "stare": "staring", "stares": "staring",
            "sit": "sitting", "sits": "sitting",
            "dance": "dancing", "dances": "dancing",
            "open": "opening", "opens": "opening",
            "bite": "biting", "bites": "biting",
            "yawn": "yawning", "yawns": "yawning",
        }
        queries.extend(["2D cartoon cat animation", "cartoon kitten funny animation"])
        for sentence in re.split(r"(?<=[.!?])\\s+", script or ""):
            tokens = re.findall(r"[a-zA-Z]+", sentence.lower())
            actions = list(dict.fromkeys(action_terms[t] for t in tokens if t in action_terms))
            for action in actions[:2]:
                queries.extend([f"cartoon cat {action}", f"2D animated cat {action}"])
        queries.extend(["funny cats", "cats being funny", "cats playing", "funny kittens"])""",
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

print("Funny Kitties: narration-action cartoon footage search enabled.")

print("Funny Kitties Groq preparation completed.")


# Funny Kitties only: replace stock footage with scene-by-scene 2D cartoon art.
(ROOT / "src/funny_kitties_visuals.py").write_text(r"""
import os
import re
import subprocess
from pathlib import Path

import requests

MODEL = "@cf/black-forest-labs/flux-1-schnell"
STYLE = (
    "Vibrant, high-energy 2D animated comedy cartoon frame with the feel of "
    "a modern irreverent sci-fi cartoon: bold dark outlines, exaggerated expressive "
    "faces, unusual funny proportions, dynamic poses, punchy visual comedy, "
    "saturated electric colors, colorful lighting, richly colored environments "
    "and imaginative background details. Every scene must be visually colorful: "
    "use vivid combinations of orange, cyan, purple, lime, pink, yellow and blue; "
    "avoid grey, beige, washed-out or dull palettes. Cats are original colorful "
    "cartoon characters with distinctive bright fur colors and markings, not "
    "realistic animals. Keep the main cat's chosen bright colors and markings "
    "consistent throughout this one video, while allowing other cats and objects "
    "to have varied vivid colors. Flat 2D hand-drawn animation look with simple "
    "cel shading, clean readable shapes, vertical 9:16 composition. No text, "
    "captions, letters, speech bubbles, watermark or logo. Not photorealistic, "
    "not live action, not 3D CGI."
)

def _sentences(script):
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", script or "") if x.strip()]
    if not parts:
        raise RuntimeError("Funny Kitties: script contains no usable scenes.")
    # Keep scene count manageable while preserving the story order.
    if len(parts) > 9:
        groups = []
        step = len(parts) / 9
        for i in range(9):
            start = round(i * step)
            end = round((i + 1) * step)
            groups.append(" ".join(parts[start:end]))
        return groups
    return parts

def _generate_image(prompt, path, token, account_id, seed):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "prompt": prompt[:2000],
            "steps": 4,
        },
        timeout=180,
    )
    if not response.ok:
        raise RuntimeError(f"Cloudflare image API {response.status_code}: {response.text[:700]}")
    payload = response.json()
    image_b64 = (payload.get("result") or {}).get("image")
    if not image_b64:
        raise RuntimeError(f"Cloudflare returned no image data: {str(payload)[:500]}")
    import base64
    path.write_bytes(base64.b64decode(image_b64))
    if not path.exists() or path.stat().st_size < 1000:
        raise RuntimeError("Generated cartoon image is empty or invalid.")

def fetch_funny_kitties_scenes(topic, script, out_dir, count=8, orientation="portrait"):
    token = os.environ.get("PEXELS_API_KEY", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not token:
        raise RuntimeError("PEXELS_API_KEY is missing. Add your Cloudflare Workers AI token to this existing secret.")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is missing. Add your Cloudflare Account ID as a GitHub secret.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenes = _sentences(script)
    if len(scenes) > count:
        scenes = scenes[:count]
    print(f"Funny Kitties: generating {len(scenes)} bespoke 2D cartoon scenes with {MODEL}.")
    clips = []
    for i, sentence in enumerate(scenes):
        image_path = out / f"scene_{i:02d}.png"
        clip_path = out / f"clip_{i:02d}.mp4"
        prompt = (
            f"Create one finished frame from a coherent, colorful 2D animated comedy "
            f"short. SCENE ACTION: {sentence}. STORY CONTEXT: {topic}. "
            f"Illustrate exactly what the narration says: the correct character, "
            f"action, location, objects and reaction. Do not invent a different event "
            f"or show a generic cat sitting. Make the action instantly understandable "
            f"and visually funny, with expressive faces, dynamic staging and a vivid "
            f"colorful environment. Maintain the same main cat design used in the "
            f"other scenes of this video. One scene only, no collage or panels. "
            f"{STYLE}"
        )
        _generate_image(prompt, image_path, token, account_id, 7300 + i)
        print(f"Funny Kitties: generated scene {i+1}/{len(scenes)} for: {sentence}")
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "zoompan=z='min(zoom+0.0008,1.06)':d=150:s=1080x1920:fps=30,"
            "format=yuv420p"
        )
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image_path),
            "-vf", vf, "-t", "5", "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", str(clip_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size < 10000:
            raise RuntimeError(f"Could not animate cartoon scene {i+1}: {result.stdout[-1200:]}")
        clips.append(str(clip_path))
    if not clips:
        raise RuntimeError("Funny Kitties: no cartoon scenes were generated.")
    print(f"Funny Kitties: ready {len(clips)} animated 2D cartoon scene clips.")
    return clips
""".lstrip(), encoding="utf-8")

replace(
    "src/main.py",
    "from src.fetch_stock import fetch_clips_for_topic",
    "from src.fetch_stock import fetch_clips_for_topic\nfrom src.funny_kitties_visuals import fetch_funny_kitties_scenes",
    1,
)

replace(
    "src/main.py",
    """            clip_paths = fetch_clips_for_topic(
                topic,
                str(clips_dir),
                count=clip_count,
                orientation=(
                    "portrait"
                    if is_short
                    else "landscape"
                ),
                script=script_text,
            )""",
    """            if channel_id == "funny_cats":
                clip_paths = fetch_funny_kitties_scenes(
                    topic,
                    script_text,
                    str(clips_dir),
                    count=9 if is_short else 30,
                    orientation="portrait" if is_short else "landscape",
                )
            else:
                clip_paths = fetch_clips_for_topic(
                    topic,
                    str(clips_dir),
                    count=clip_count,
                    orientation=(
                        "portrait"
                        if is_short
                        else "landscape"
                    ),
                )""",
    1,
)

print("Funny Kitties: Cloudflare FLUX scene-by-scene cartoon pipeline installed.")
