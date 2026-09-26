"""
generate_script.py

Generates scripts for the Autotube channels using Google's current
google-genai SDK.

IMPORTANT:
- Uses client.chats.create() + chat.send_message()
- Does NOT use client.models.generate_content() for text generation.
- This avoids the AFC/direct model-call warning seen with newer SDKs.
- Falls back to template scripts if Gemini is unavailable.
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


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

try:
    from google import genai as genai_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


GEMINI_MODEL = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# Token budgeting
# ---------------------------------------------------------------------------

TOKENS_PER_WORD = 2.2
MIN_OUTPUT_TOKENS = 512
TOKEN_SAFETY_MARGIN = 200


def _max_output_tokens_for(words_target: int) -> int:
    return max(
        MIN_OUTPUT_TOKENS,
        int(words_target * TOKENS_PER_WORD) + TOKEN_SAFETY_MARGIN,
    )


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (written in Devanagari script, not transliterated/Roman Hindi)",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_channel_config(channel_id: str) -> dict:
    path = ROOT / "channels" / f"{channel_id}.yaml"

    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Topic state
# ---------------------------------------------------------------------------

def _used_topics_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_used_topics.json"


def get_used_topics(channel_id: str) -> set:
    path = _used_topics_path(channel_id)

    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            return set()

    return set()


def mark_topic_used(channel_id: str, topic: str) -> None:
    used = get_used_topics(channel_id)
    used.add(topic)

    _used_topics_path(channel_id).write_text(
        json.dumps(sorted(used), ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Long-form -> Short recap state
# ---------------------------------------------------------------------------

def _todays_longform_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_todays_longform.json"


def save_todays_longform_topic(channel_id: str, topic: str) -> None:
    """
    Records today's long-form topic so one Short can reuse it as a recap.
    """

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    _todays_longform_path(channel_id).write_text(
        json.dumps(
            {
                "date": today,
                "topic": topic,
                "used_for_short": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def get_recap_topic_for_short(channel_id: str) -> str | None:
    """
    Returns today's long-form topic if it has not already been used
    for a recap Short today.
    """

    path = _todays_longform_path(channel_id)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if data.get("date") != today:
        return None

    if data.get("used_for_short"):
        return None

    topic = data.get("topic")

    if not topic:
        return None

    data["used_for_short"] = True

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )
    )

    return topic


# ---------------------------------------------------------------------------
# Gemini Chat generation
# ---------------------------------------------------------------------------

def _generate_with_token_budget(
    client,
    prompt: str,
    max_output_tokens: int,
):
    """
    Generate text using the google-genai Chat API.

    IMPORTANT:
    This intentionally uses:

        client.chats.create()
        chat.send_message()

    and never:


    The Chat API is used throughout this file for text generation.
    """

    try:
        from google.genai import types

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
        )

        chat = client.chats.create(
            model=GEMINI_MODEL,
            config=generation_config,
        )

        response = chat.send_message(prompt)

    except Exception as first_error:
        print(
            "WARNING: Gemini Chat generation with explicit token config "
            f"failed ({first_error}); retrying without explicit config."
        )

        chat = client.chats.create(
            model=GEMINI_MODEL,
        )

        response = chat.send_message(prompt)

    finish_reason = None

    try:
        finish_reason = response.candidates[0].finish_reason
    except Exception:
        pass

    if finish_reason and "MAX_TOKENS" in str(finish_reason).upper():
        print(
            "WARNING: Gemini response was truncated by "
            f"max_output_tokens={max_output_tokens}. "
            "The generated script may be shorter than the target."
        )

    return response


# ---------------------------------------------------------------------------
# Topic generation
# ---------------------------------------------------------------------------

def _generate_more_topics(
    config: dict,
    existing: list,
    count: int = 20,
) -> list:
    """
    Ask Gemini for fresh topic ideas.
    """

    if not (
        GEMINI_AVAILABLE
        and os.environ.get("GEMINI_API_KEY")
    ):
        return []

    try:
        client = genai_client.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        existing_sample = "\n".join(
            f"- {topic}"
            for topic in existing[-40:]
        )

        prompt = f"""
You generate topic ideas for a faceless YouTube channel.

Channel: {config['display_name']}
Niche: {config['niche']}
Tone: {config['tone']}

Here are topics already covered. Do NOT repeat these or close variations:

{existing_sample}

Generate {count} brand new topic ideas for this channel.

Each topic must:
- Be a single line.
- Be specific enough to script a video from.
- Be genuinely different from the existing topics.
- Be interesting to viewers.

No numbering.
No markdown.
No quotes.

Just one topic per line.
"""

        response = _generate_with_token_budget(
            client,
            prompt,
            max_output_tokens=800,
        )

        lines = [
            re.sub(
                r"^[\d\.\-\)\s]+",
                "",
                line,
            ).strip()
            for line in response.text.splitlines()
        ]

        new_topics = [
            line
            for line in lines
            if line
            and line not in existing
        ]

        return new_topics

    except Exception as e:
        print(
            "WARNING: topic auto-generation failed "
            f"({e}); will loop existing topics instead."
        )

        return []


def pick_next_topic(
    channel_id: str,
    config: dict,
) -> str:

    topics_file = ROOT / config["topics_seed_file"]

    all_topics = [
        line.strip()
        for line in topics_file.read_text().splitlines()
        if line.strip()
    ]

    used = get_used_topics(channel_id)

    unused = [
        topic
        for topic in all_topics
        if topic not in used
    ]

    if not unused:

        new_topics = _generate_more_topics(
            config,
            all_topics,
        )

        if new_topics:

            with topics_file.open("a") as f:
                f.write(
                    "\n"
                    + "\n".join(new_topics)
                    + "\n"
                )

            print(
                f"Added {len(new_topics)} new topics "
                f"to {topics_file.name}"
            )

            unused = new_topics

        else:
            unused = all_topics

    if not unused:
        raise RuntimeError(
            f"No topics available for channel '{channel_id}'."
        )

    topic = random.choice(unused)

    mark_topic_used(
        channel_id,
        topic,
    )

    return topic


# ---------------------------------------------------------------------------
# Language / voice
# ---------------------------------------------------------------------------

def pick_language(config: dict) -> str:
    languages = config.get("languages") or ["en"]

    return random.choice(languages)


def _voice_for_language(
    config: dict,
    language: str,
) -> str:

    voices = config.get("voices") or {}

    voice = (
        voices.get(language)
        or config.get("voice")
    )

    if not voice:
        raise ValueError(
            f"No TTS voice configured for language "
            f"'{language}' in this channel's yaml."
        )

    return voice


# ---------------------------------------------------------------------------
# Script cleaning
# ---------------------------------------------------------------------------

def _clean_script_text(text: str) -> str:
    """
    Remove markdown and stage directions from narration.
    """

    text = re.sub(
        r"\*+",
        "",
        text,
    )

    text = re.sub(
        r"\[.*?\]",
        "",
        text,
    )

    text = re.sub(
        r"\(.*?\)",
        "",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Title/script parser
# ---------------------------------------------------------------------------

def _parse_titled_response(
    text: str,
    fallback_topic: str,
) -> dict:

    title_match = re.search(
        r"TITLE:\s*(.+)",
        text,
    )

    script_match = re.search(
        r"SCRIPT:\s*(.*)",
        text,
        re.DOTALL,
    )

    title = (
        title_match.group(1).strip()
        if title_match
        else fallback_topic
    )

    script_raw = (
        script_match.group(1).strip()
        if script_match
        else text
    )

    return {
        "title": title,
        "script": _clean_script_text(script_raw),
    }


# ---------------------------------------------------------------------------
# Main Cosmic Curious Gemini generation
# ---------------------------------------------------------------------------

def generate_with_gemini(
    topic: str,
    config: dict,
    length_seconds: int,
    language: str,
) -> dict:

    client = genai_client.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    if length_seconds <= 45:
        words_target = 115
    elif length_seconds <= 55:
        words_target = 135
    else:
        # Scale properly for real long-form durations instead of capping
        # at a Short-sized word count -- ~140 words/minute is a
        # reasonable narrated pace (accounts for natural pauses and the
        # slight pacing slowdown applied in tts.py). Without this
        # scaling, EVERY long-form video was getting the same ~150-word
        # target regardless of video_length_seconds -- confirmed by
        # real published videos consistently landing at ~100-150 words
        # even when video_length_seconds was set to 300 (5 minutes).
        words_target = round(length_seconds / 60 * 140)

    language_name = LANGUAGE_NAMES.get(
        language,
        language,
    )

 prompt = f"""
You generate topic ideas for a faceless YouTube channel.

CHANNEL NICHE:
Animals and wildlife facts.

CHANNEL TONE:
Curious, surprising, energetic, conversational, and easy to understand.

Generate fresh YouTube topic ideas ONLY about animals and wildlife.

TOPIC TYPES:
- Strange animal abilities
- Incredible animal senses
- Animal behavior
- Survival skills
- Unusual adaptations
- Record-breaking animals
- Dangerous animals
- Cute but surprising animal facts
- Deep-sea animals
- Rare animals
- Animal intelligence
- Predator and prey facts
- Unexpected animal relationships

IMPORTANT:
- Do NOT generate topics about space.
- Do NOT generate topics about planets, stars, galaxies, black holes, astronomy, physics, or the universe.
- Do NOT generate generic science topics unrelated to animals.
- Do NOT repeat topics already used.
- Use real, well-established facts.
- Make every topic interesting enough to become a YouTube Short.
- Avoid boring school-style topics.
- Make the ideas sound surprising and clickable without being misleading.

CHANNEL:
{config['display_name']}

NICHE:
{config['niche']}

TONE:
{config['tone']}

Generate a list of fresh topic ideas.
"""

        response = _generate_with_token_budget(
            client,
            prompt,
            max_output_tokens=800,
        )

        lines = [
            re.sub(
                r"^[\d\.\-\)\s]+",
                "",
                line,
            ).strip()
            for line in response.text.splitlines()
        ]

        new_topics = [
            line
            for line in lines
            if line
            and line not in existing
        ]

        return new_topics

    except Exception as e:

        print(
            "WARNING: kids topic auto-generation failed "
            f"({e}); will loop existing topics instead."
        )

        return []


def pick_next_kids_topic(
    channel_id: str,
    config: dict,
    content_type: str,
) -> str:

    topics_file = _kids_seed_path(
        config,
        content_type,
    )

    dedupe_key = _kids_dedupe_key(
        channel_id,
        content_type,
    )

    all_topics = [
        line.strip()
        for line in topics_file.read_text().splitlines()
        if line.strip()
    ]

    used = get_used_topics(
        dedupe_key
    )

    unused = [
        topic
        for topic in all_topics
        if topic not in used
    ]

    if not unused:

        new_topics = _generate_more_kids_topics(
            config,
            content_type,
            all_topics,
        )

        if new_topics:

            with topics_file.open("a") as f:
                f.write(
                    "\n"
                    + "\n".join(new_topics)
                    + "\n"
                )

            print(
                f"Added {len(new_topics)} new "
                f"{content_type} themes to "
                f"{topics_file.name}"
            )

            unused = new_topics

        else:

            unused = all_topics

    if not unused:
        raise RuntimeError(
            f"No topics available for kids "
            f"content type '{content_type}'."
        )

    topic = random.choice(
        unused
    )

    mark_topic_used(
        dedupe_key,
        topic,
    )

    return topic


# ---------------------------------------------------------------------------
# Kids Gemini generation
# ---------------------------------------------------------------------------

def generate_kids_script_with_gemini(
    topic: str,
    config: dict,
    content_type: str,
    mascot_name: str,
    length_seconds: int,
    language: str,
) -> dict:

    client = genai_client.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    words_target = int(
        length_seconds * 2.2
    )

    language_name = LANGUAGE_NAMES.get(
        language,
        language,
    )

    if content_type == "rhyme":

        content_instructions = f"""
Write an ORIGINAL Hindi rhyme/poem for young children ages 2-6,
themed around:

{topic}

CRITICAL:

This must be a completely original composition.

Do NOT reproduce, translate, or closely imitate any existing,
traditional, or copyrighted nursery rhyme, song, or poem.

Write it purely in Hindi using Devanagari script.

Use simple everyday words a toddler already knows.

Keep a clear, consistent rhythm and rhyme scheme.

Include a short repeated refrain or chorus.

Include at least one simple action kids can copy, such as:
clap, jump, sway.

The character {mascot_name} should be the one singing or leading it,
mentioned warmly by name at least once.

Keep the mood joyful and gentle.
"""

    elif content_type == "story":

        content_instructions = f"""
Write an ORIGINAL short moral story for young children ages 3-7,
themed around:

{topic}

The story should feature {mascot_name}.

Teach one simple positive lesson such as:
sharing, kindness, honesty, or trying again.

Show the lesson through what happens rather than lecturing.

Only state the lesson gently at the very end.

Language style:

Natural Hinglish code-mixing commonly heard in Indian children's
content.

Mostly simple Hindi with a handful of everyday English words
mixed naturally.

Keep sentences short and simple.

Nothing scary, violent, or sad.

Any conflict should be gentle and resolved warmly.
"""

    else:

        content_instructions = f"""
Write an ORIGINAL short learning segment for young children ages 2-6,
teaching:

{topic}

{mascot_name} should teach directly to the viewer.

Use a warm and encouraging tone.

Language style:

Natural bilingual teaching commonly used in Indian children's
educational content.

Introduce concepts in Hindi and reinforce them with simple English
equivalents.

Use call-and-response phrasing such as:

"bolo mere saath..."

Keep it repetitive and simple.
"""

    prompt = f"""
You are writing a script for a children's YouTube video.

Channel:
{config['display_name']}

Content type:
{content_type}

Language:
{language_name}

{content_instructions}

Respond in EXACTLY this format and nothing else:

TITLE: <a warm, simple title in {language_name}, under 90 characters>

SCRIPT:
<spoken narration only>

No markdown.
No stage directions.
No visual cues.
No commentary.

Target script length:
approximately {words_target} words.

End on a warm, gentle closing line.
Do not end abruptly.
"""

    max_tokens = _max_output_tokens_for(
        words_target
    )

    response = _generate_with_token_budget(
        client,
        prompt,
        max_tokens,
    )

    return _parse_titled_response(
        response.text,
        fallback_topic=topic,
    )


# ---------------------------------------------------------------------------
# Kids template fallback
# ---------------------------------------------------------------------------

def generate_kids_template(
    topic: str,
    config: dict,
    content_type: str,
    mascot_name: str,
    language: str,
) -> dict:

    if content_type == "rhyme":

        script = (
            f"चलो सब मिलकर गाएं, {mascot_name} के साथ। "
            f"आज की कहानी है {topic} के बारे में। "
            f"ताली बजाओ, संग गाओ, मज़ा करो, हाँ! "
            f"यही तो है हमारी प्यारी सी धुन, "
            f"फिर मिलेंगे, बाय बाय!"
        )

        title = (
            f"{mascot_name} की मस्ती भरी कविता"
        )

    elif content_type == "story":

        script = (
            f"एक बार की बात है, {mascot_name} नाम का "
            f"एक प्यारा दोस्त था। "
            f"एक दिन उसे पता चला {topic} के बारे में "
            f"एक important lesson। "
            f"उसने सीखा कि हमेशा kind और honest रहना चाहिए। "
            f"अंत में सब दोस्त बहुत खुश हुए। "
            f"The end!"
        )

        title = (
            f"{mascot_name} की एक प्यारी कहानी"
        )

    else:

        script = (
            f"नमस्ते दोस्तों! मैं हूँ {mascot_name}। "
            f"आज हम सीखेंगे {topic}। "
            f"बोलो मेरे साथ! "
            f"बहुत बढ़िया! "
            f"अब आप भी जान गए। "
            f"Great job, दोस्तों! "
            f"फिर मिलेंगे अगली सीख के साथ!"
        )

        title = (
            f"{mascot_name} के साथ सीखो: {topic}"
        )

    return {
        "title": title,
        "script": script,
    }


# ---------------------------------------------------------------------------
# Public kids API
# ---------------------------------------------------------------------------

def generate_kids_script(
    channel_id: str,
    is_short: bool = False,
) -> dict:

    config = load_channel_config(
        channel_id
    )

    content_type = pick_content_type(
        config
    )

    topic = pick_next_kids_topic(
        channel_id,
        config,
        content_type,
    )

    mascot = config["mascot_map"][
        content_type
    ]

    mascot_name = config["mascot_names"][
        mascot
    ]

    # Rhymes use Hindi only so the rhyme scheme remains consistent.
    if (
        content_type == "rhyme"
        and config.get(
            "rhymes_hindi_only",
            True,
        )
    ):
        language = "hi"

    else:
        language = pick_language(
            config
        )

    length = (
        config["short_length_seconds"]
        if is_short
        else config["video_length_seconds"]
    )

    if (
        GEMINI_AVAILABLE
        and os.environ.get("GEMINI_API_KEY")
    ):

        try:

            generated = generate_kids_script_with_gemini(
                topic,
                config,
                content_type,
                mascot_name,
                length,
                language,
            )

        except Exception as e:

            print(
                f"WARNING: Gemini call failed ({e}); "
                "using template fallback."
            )

            generated = generate_kids_template(
                topic,
                config,
                content_type,
                mascot_name,
                language,
            )

    else:

        generated = generate_kids_template(
            topic,
            config,
            content_type,
            mascot_name,
            language,
        )

    return {
        "channel_id": channel_id,
        "topic": topic,
        "title": generated["title"],
        "script": generated["script"],
        "language": language,
        "voice": _voice_for_language(
            config,
            language,
        ),
        "is_short": is_short,
        "content_type": content_type,
        "mascot": mascot,
        "mascot_name": mascot_name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "channel_id"
    )

    parser.add_argument(
        "--short",
        action="store_true",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Path to write JSON output",
    )

    args = parser.parse_args()

    result = generate_script(
        args.channel_id,
        is_short=args.short,
    )

    output = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    if args.out:
        Path(args.out).write_text(
            output
        )

    print(output)