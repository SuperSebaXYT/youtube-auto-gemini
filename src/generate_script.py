"""
generate_script.py

Generates YouTube scripts for the existing Autotube pipeline.
The main channel is Animals & Wildlife. Compatibility helpers for the
existing kids pipeline are intentionally preserved.
"""

import os
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.trend_scout import get_trending_topic
from src.daily_brain_topics import get_daily_brain_topic

try:
    from google import genai as genai_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_MODEL = "gemini-3.5-flash-lite"
TOKENS_PER_WORD = 2.2
MIN_OUTPUT_TOKENS = 512
TOKEN_SAFETY_MARGIN = 200

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (written in Devanagari script, not Roman Hindi)",
}

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)


def _max_output_tokens_for(words_target: int) -> int:
    return max(MIN_OUTPUT_TOKENS, int(words_target * TOKENS_PER_WORD) + TOKEN_SAFETY_MARGIN)


def load_channel_config(channel_id: str) -> dict:
    path = ROOT / "channels" / f"{channel_id}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _used_topics_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_used_topics.json"


def get_used_topics(channel_id: str) -> set:
    path = _used_topics_path(channel_id)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def mark_topic_used(channel_id: str, topic: str) -> None:
    used = get_used_topics(channel_id)
    used.add(topic)
    _used_topics_path(channel_id).write_text(
        json.dumps(sorted(used), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _todays_longform_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_todays_longform.json"


def save_todays_longform_topic(channel_id: str, topic: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _todays_longform_path(channel_id).write_text(
        json.dumps(
            {"date": today, "topic": topic, "used_for_short": False},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_recap_topic_for_short(channel_id: str) -> str | None:
    path = _todays_longform_path(channel_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data.get("date") != today or data.get("used_for_short"):
        return None
    topic = data.get("topic")
    if not topic:
        return None
    data["used_for_short"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return topic


def _generate_with_token_budget(client, prompt: str, max_output_tokens: int):
    try:
        from google.genai import types
        generation_config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)
        chat = client.chats.create(model=GEMINI_MODEL, config=generation_config)
        response = chat.send_message(prompt)
    except Exception as first_error:
        print(f"WARNING: Gemini configured chat call failed ({first_error}); retrying without explicit config.")
        chat = client.chats.create(model=GEMINI_MODEL)
        response = chat.send_message(prompt)

    try:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "MAX_TOKENS" in str(finish_reason).upper():
            print(f"WARNING: Gemini response hit max_output_tokens={max_output_tokens}.")
    except Exception:
        pass
    return response


def _generate_more_topics(config: dict, existing: list, count: int = 20) -> list:
    if not (GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY")):
        return []
    try:
        client = genai_client.Client(api_key=os.environ["GEMINI_API_KEY"])
        existing_sample = "\n".join(f"- {topic}" for topic in existing[-40:])
        prompt = f"""
Generate {count} fresh topics for a faceless YouTube channel.

CHANNEL: {config['display_name']}
NICHE: Animals & wildlife facts
TONE: {config['tone']}

Already used:
{existing_sample}

Every topic must be about real animals or wildlife. Use specific,
surprising subjects such as animal abilities, senses, behavior,
survival, adaptations, intelligence, predators, prey, rare animals,
deep-sea animals, or unusual animal relationships.

Do NOT generate space, planets, stars, galaxies, black holes,
astronomy, NASA, generic physics, or unrelated science.
Do not repeat existing topics or close variations.

One topic per line. No numbering, markdown, or quotes.
"""
        response = _generate_with_token_budget(client, prompt, 800)
        existing_lower = {x.strip().lower() for x in existing}
        lines = [re.sub(r"^[\d\.\-\)\s]+", "", x).strip() for x in response.text.splitlines()]
        return [x for x in lines if x and x.lower() not in existing_lower]
    except Exception as e:
        print(f"WARNING: topic auto-generation failed ({e}); will use existing topics.")
        return []


def pick_next_topic(channel_id: str, config: dict) -> str:
    topics_file = ROOT / config["topics_seed_file"]
    all_topics = [x.strip() for x in topics_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    used = get_used_topics(channel_id)
    unused = [x for x in all_topics if x not in used]

    if not unused:
        new_topics = _generate_more_topics(config, all_topics)
        if new_topics:
            with topics_file.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_topics) + "\n")
            unused = new_topics
        else:
            unused = all_topics

    if not unused:
        raise RuntimeError(f"No topics available for channel '{channel_id}'.")
    topic = random.choice(unused)
    mark_topic_used(channel_id, topic)
    return topic


def pick_language(config: dict) -> str:
    return random.choice(config.get("languages") or ["en"])


def _voice_for_language(config: dict, language: str) -> str:
    voices = config.get("voices") or {}
    voice = voices.get(language) or config.get("voice")
    if not voice:
        raise ValueError(f"No TTS voice configured for language '{language}'.")
    return voice


def _clean_script_text(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_titled_response(text: str, fallback_topic: str) -> dict:
    title_match = re.search(r"TITLE:\s*(.+)", text)
    script_match = re.search(r"SCRIPT:\s*(.*)", text, re.DOTALL)
    return {
        "title": title_match.group(1).strip() if title_match else fallback_topic,
        "script": _clean_script_text(script_match.group(1).strip() if script_match else text),
    }


def generate_with_gemini(topic: str, config: dict, length_seconds: int, language: str) -> dict:
    client = genai_client.Client(api_key=os.environ["GEMINI_API_KEY"])
    if length_seconds <= 45:
        words_target = 115
    elif length_seconds <= 55:
        words_target = 135
    else:
        words_target = round(length_seconds / 60 * 140)

    language_name = LANGUAGE_NAMES.get(language, language)
    prompt = f"""
You are the lead writer for a faceless YouTube channel called "{config['display_name']}".

NICHE: Animals and wildlife facts.
TONE: Curious, surprising, energetic, punchy, conversational — like telling a friend an amazing animal fact.
TOPIC: {topic}

Write a highly engaging voiceover about this exact animal/wildlife topic.

CONTENT RULES:
- The entire script must be about animals or wildlife.
- Use real, well-established facts.
- Never invent statistics, discoveries, quotes, experiments, behaviors, or abilities.
- Never exaggerate an animal's abilities or present speculation as fact.
- Do NOT write about space, planets, stars, galaxies, black holes, astronomy, NASA, the universe, or unrelated physics.
- Keep the focus on the specific animal, species, behavior, adaptation, ability, survival strategy, or relationship in the topic.

STRUCTURE:
1. Open with the strangest or most surprising consequence, question, or image connected to the animal.
2. Give only enough setup to understand the animal and situation.
3. Reveal interesting facts step by step, increasing curiosity.
4. Save a genuinely surprising fact for the payoff.
5. End with a memorable line directly connected to the animal.

STYLE:
- Natural spoken language suitable for text-to-speech.
- Short sentences mixed with occasional longer ones.
- No academic-paper language, filler, generic listicles, motivational lessons, emojis, stage directions, visual labels, or markdown.
- Do not begin by simply repeating the topic.
- Do not mention AI.
- Do not say subscribe, like, or follow.

LANGUAGE: Write entirely in {language_name}.
If Hindi is requested, use Devanagari, not Romanized Hindi.

LENGTH: approximately {words_target} words. Do not pad with repetition.

OUTPUT EXACTLY:
TITLE: <punchy YouTube title under 90 characters>

SCRIPT:
<spoken narration only>
"""
    response = _generate_with_token_budget(client, prompt, _max_output_tokens_for(words_target))
    result = _parse_titled_response(response.text, topic)
    result["title"] = result["title"].strip().strip('"').strip("'")
    return result


def generate_with_template(topic: str, config: dict, length_seconds: int, language: str) -> dict:
    return {
        "title": topic.strip(),
        "script": (
            f"{topic}. The surprising part is what happens next. "
            "This animal has a remarkable way of surviving, behaving, "
            "or adapting to its environment. And the strangest part is "
            "that this is a real behavior found in the animal world."
        ),
    }


def generate_script(channel_id: str, is_short: bool = False, forced_topic: str | None = None) -> dict:
    config = load_channel_config(channel_id)
    length = config["short_length_seconds"] if is_short else config["video_length_seconds"]
    language = pick_language(config)
    topic = forced_topic

    if not topic:
        try:
            topic = get_daily_brain_topic(channel_id, config, is_short)
        except Exception as e:
            print(f"WARNING: Daily Brain topic selection failed ({e}); using fallback selection.")

    if not topic:
        try:
            topic = get_trending_topic(channel_id, config)
        except Exception as e:
            print(f"WARNING: Trend Scout topic selection failed ({e}); using static topics.")

    if not topic:
        topic = pick_next_topic(channel_id, config)

    if forced_topic:
        mark_topic_used(channel_id, topic)

    generated = None
    if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        try:
            generated = generate_with_gemini(topic, config, length, language)
        except Exception as e:
            print(f"WARNING: Gemini script generation failed ({e}); using template fallback.")

    if not generated:
        generated = generate_with_template(topic, config, length, language)

    return {
        "channel_id": channel_id,
        "topic": topic,
        "title": generated["title"],
        "script": generated["script"],
        "language": language,
        "voice": _voice_for_language(config, language),
        "is_short": is_short,
    }


# ---------------------------------------------------------------------------
# Kids compatibility helpers
# ---------------------------------------------------------------------------

def _kids_seed_path(config: dict, content_type: str) -> Path:
    files = config.get("kids_topics_files") or {}
    path_value = files.get(content_type) or config.get(f"{content_type}_topics_seed_file") or config.get("kids_topics_seed_file")
    if not path_value:
        raise KeyError(f"No kids topic seed file configured for '{content_type}'.")
    return ROOT / path_value


def _kids_dedupe_key(channel_id: str, content_type: str) -> str:
    return f"{channel_id}_kids_{content_type}"


def pick_content_type(config: dict) -> str:
    choices = config.get("kids_content_types") or config.get("content_types") or ["lesson", "story", "rhyme"]
    if isinstance(choices, dict):
        choices = list(choices.keys())
    return random.choice(list(choices)) if choices else "lesson"


def _generate_more_kids_topics(config: dict, content_type: str, existing: list, count: int = 20) -> list:
    if not (GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY")):
        return []
    try:
        client = genai_client.Client(api_key=os.environ["GEMINI_API_KEY"])
        existing_sample = "\n".join(f"- {x}" for x in existing[-40:])
        prompt = f"""
Generate {count} original children's YouTube topic ideas.
Channel: {config.get('display_name', 'Kids channel')}
Content type: {content_type}
Already used:
{existing_sample}

Each topic must fit the requested content type and be suitable for young children.
Do not copy existing topics. Do not use copyrighted song/story titles.
One topic per line. No numbering, markdown, or quotes.
"""
        response = _generate_with_token_budget(client, prompt, 700)
        existing_lower = {x.lower() for x in existing}
        lines = [re.sub(r"^[\d\.\-\)\s]+", "", x).strip() for x in response.text.splitlines()]
        return [x for x in lines if x and x.lower() not in existing_lower]
    except Exception as e:
        print(f"WARNING: kids topic auto-generation failed ({e}); looping existing topics.")
        return []


def pick_next_kids_topic(channel_id: str, config: dict, content_type: str) -> str:
    topics_file = _kids_seed_path(config, content_type)
    dedupe_key = _kids_dedupe_key(channel_id, content_type)
    all_topics = [x.strip() for x in topics_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    used = get_used_topics(dedupe_key)
    unused = [x for x in all_topics if x not in used]

    if not unused:
        new_topics = _generate_more_kids_topics(config, content_type, all_topics)
        if new_topics:
            with topics_file.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_topics) + "\n")
            unused = new_topics
        else:
            unused = all_topics

    if not unused:
        raise RuntimeError(f"No topics available for kids content type '{content_type}'.")
    topic = random.choice(unused)
    mark_topic_used(dedupe_key, topic)
    return topic


def generate_kids_script_with_gemini(topic: str, config: dict, content_type: str, mascot_name: str, length_seconds: int, language: str) -> dict:
    client = genai_client.Client(api_key=os.environ["GEMINI_API_KEY"])
    words_target = int(length_seconds * 2.2)
    language_name = LANGUAGE_NAMES.get(language, language)

    if content_type == "rhyme":
        content_instructions = f"""
Write an ORIGINAL Hindi rhyme/poem for children ages 2-6 about:
{topic}
It must be completely original and not reproduce or closely imitate an existing nursery rhyme, song, or poem.
Use Devanagari, simple words, clear rhythm, a short refrain, and one easy action kids can copy.
{mascot_name} should lead it. Keep it joyful and gentle.
"""
    elif content_type == "story":
        content_instructions = f"""
Write an ORIGINAL short moral story for children ages 3-7 about:
{topic}
Feature {mascot_name}. Teach one simple positive lesson through the story.
Use short, simple sentences and gentle conflict. Nothing scary, violent, or sad.
Natural Hinglish code-mixing is allowed.
"""
    else:
        content_instructions = f"""
Write an ORIGINAL short learning segment for children ages 2-6 teaching:
{topic}
{mascot_name} teaches directly to the viewer using warm call-and-response phrasing.
Use simple bilingual teaching with Hindi and everyday English.
"""

    prompt = f"""
You are writing a children's YouTube video script.
Channel: {config['display_name']}
Content type: {content_type}
Language: {language_name}
{content_instructions}

Respond exactly:
TITLE: <warm title under 90 characters>

SCRIPT:
<spoken narration only>

No markdown, stage directions, visual cues, or commentary.
Target approximately {words_target} words. End warmly.
"""
    response = _generate_with_token_budget(client, prompt, _max_output_tokens_for(words_target))
    return _parse_titled_response(response.text, topic)


def generate_kids_template(topic: str, config: dict, content_type: str, mascot_name: str, language: str) -> dict:
    if content_type == "rhyme":
        return {"title": f"{mascot_name} की मस्ती भरी कविता", "script": f"चलो सब मिलकर गाएं, {mascot_name} के साथ। आज की कहानी है {topic} के बारे में। ताली बजाओ, संग गाओ, मज़ा करो, हाँ! फिर मिलेंगे, बाय बाय!"}
    if content_type == "story":
        return {"title": f"{mascot_name} की एक प्यारी कहानी", "script": f"एक बार की बात है, {mascot_name} नाम का एक प्यारा दोस्त था। एक दिन उसे पता चला {topic} के बारे में एक important lesson. उसने सीखा कि हमेशा kind और honest रहना चाहिए। अंत में सब दोस्त बहुत खुश हुए। The end!"}
    return {"title": f"{mascot_name} के साथ सीखो: {topic}", "script": f"नमस्ते दोस्तों! मैं हूँ {mascot_name}। आज हम सीखेंगे {topic}। बोलो मेरे साथ! बहुत बढ़िया! अब आप भी जान गए। Great job, दोस्तों! फिर मिलेंगे अगली सीख के साथ!"}


def generate_kids_script(channel_id: str, is_short: bool = False) -> dict:
    config = load_channel_config(channel_id)
    content_type = pick_content_type(config)
    topic = pick_next_kids_topic(channel_id, config, content_type)
    mascot_map = config.get("mascot_map") or {}
    mascot_names = config.get("mascot_names") or {}
    mascot = mascot_map.get(content_type, next(iter(mascot_names), "mascot"))
    mascot_name = mascot_names.get(mascot, str(mascot))
    language = "hi" if content_type == "rhyme" and config.get("rhymes_hindi_only", True) else pick_language(config)
    length = config["short_length_seconds"] if is_short else config["video_length_seconds"]

    try:
        if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
            generated = generate_kids_script_with_gemini(topic, config, content_type, mascot_name, length, language)
        else:
            generated = generate_kids_template(topic, config, content_type, mascot_name, language)
    except Exception as e:
        print(f"WARNING: kids Gemini generation failed ({e}); using template fallback.")
        generated = generate_kids_template(topic, config, content_type, mascot_name, language)

    return {
        "channel_id": channel_id,
        "topic": topic,
        "title": generated["title"],
        "script": generated["script"],
        "language": language,
        "voice": _voice_for_language(config, language),
        "is_short": is_short,
        "content_type": content_type,
        "mascot": mascot,
        "mascot_name": mascot_name,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("channel_id")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--out", default=None, help="Path to write JSON output")
    args = parser.parse_args()

    result = generate_script(args.channel_id, is_short=args.short)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    print(output)