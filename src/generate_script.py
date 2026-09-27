# -*- coding: utf-8 -*-
"""
generate_script.py

Generates scripts for the Autotube channels using Google's current
google-genai SDK. Refactored 100% for animal and wildlife content.

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
    Records today's long-form animal topic so one Short can reuse it as a recap.
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
    Returns today's long-form animal topic if it has not already been used
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
# Topic generation (Animals)
# ---------------------------------------------------------------------------

def _generate_more_topics(
    config: dict,
    existing: list,
    count: int = 20,
) -> list:
    """
    Ask Gemini for fresh animal and wildlife topic ideas.
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
You generate animal and wildlife topic ideas for a faceless YouTube channel.

Channel: {config['display_name']}
Niche: {config['niche']}
Tone: {config['tone']}

Here are animal topics already covered. Do NOT repeat these or close variations:

{existing_sample}

Generate {count} brand new animal or wildlife topic ideas for this channel.

Each topic must:
- Be a single line focusing on animals, wildlife, pet behavior, or zoology.
- Be specific enough to script a video from.
- Be genuinely different from the existing topics.
- Be interesting to animal lovers.

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
            "WARNING: animal topic auto-generation failed "
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
                f"Added {len(new_topics)} new animal topics "
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
# Main Animal Wildlife Gemini generation
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
        words_target = round(length_seconds / 60 * 140)

    language_name = LANGUAGE_NAMES.get(
        language,
        language,
    )

    prompt = f"""
You are the lead wildlife and zoology writer for a premium faceless YouTube channel
called "{config['display_name']}".

CHANNEL NICHE:
{config['niche']}

CHANNEL TONE:
{config['tone']}

TOPIC:
{topic}

Write a SHORT, highly engaging animal/wildlife voiceover.

The viewer should feel:

"I had no idea animals could do that."

IMPORTANT:

This is NOT a generic motivational video.
This is NOT a random list of pet care tips.
This is NOT a school textbook introduction.

The entire script must be specifically about the supplied animal topic.

FACTUAL STANDARD:

- Use established zoological and biological knowledge whenever possible.
- Do not invent statistics, animal behaviors, quotes, or findings.
- Do not present speculation as absolute biological fact.
- Prefer concrete behavioral and physical explanations over vague descriptions.
- Avoid sensational wildlife myths that contradict established science.

RETENTION STRUCTURE:

1. HOOK

Start with the most surprising animal behavior, question, or survival adaptation.

Do NOT simply repeat the topic.

2. SETUP

Give just enough natural habitat context for the viewer to understand the animal's challenge.

3. ESCALATION

Explain the process or adaptation step by step.

Each sentence should make the situation more fascinating.

4. PAYOFF

Reveal the strangest, most surprising, or least-known evolutionary trait or consequence.

5. FINAL LINE

End on a memorable thought connected directly to the animal world.

Do NOT say:
subscribe
like
follow
in conclusion

WRITING RULES:

- Conversational spoken language suitable for nature documentaries.
- Short sentences mixed with occasional longer sentences.
- No heavy academic-paper language.
- No unnecessary jargon.
- No filler or motivational life lessons.
- Every sentence must either create curiosity, explain an animal trait, or deliver a payoff.
- Do not use emojis, stage directions, visual notes, or markdown.
- Do not mention that you are an AI.
- The narration must sound natural when read by text-to-speech.

VISUAL THINKING:

Write sentences that naturally correspond to visual wildlife moments.

LANGUAGE:

Write entirely in {language_name}.

If the requested language is Hindi, use natural modern Hindi in Devanagari,
not Romanized Hindi.

LENGTH:

Approximately {words_target} words.

Do not pad the script merely to reach the word count.
"""

    max_tokens = _max_output_tokens_for(
        words_target
    )

    response = _generate_with_token_budget(
        client,
        prompt,
        max_tokens,
    )

    result = _parse_titled_response(
        response.text,
        fallback_topic=topic,
    )

    result["title"] = (
        result["title"]
        .strip()
        .strip('"')
        .strip("'")
    )

    return result


# ---------------------------------------------------------------------------
# Emergency template fallback (Animals)
# ---------------------------------------------------------------------------

def generate_with_template(
    topic: str,
    config: dict,
    language: str,
) -> dict:

    if language == "hi":

        script = (
            f"क्या आपने कभी सोचा है कि {topic} के पीछे असल में क्या रहस्य है? "
            f"यह सवाल जितना आसान लगता है, प्रकृति की दुनिया उतनी ही दिलचस्प है। "
            f"वैज्ञानिक इस व्यवहार को समझने के लिए जीवों की आदतों का अध्ययन करते हैं। "
            f"और सबसे दिलचस्प बात यह है कि इसका जवाब हमारी सोच से कहीं ज्यादा अनोखा है।"
        )

        title = f"{topic}"

    else:

        script = (
            f"{topic}. "
            f"The surprising part is how this animal behaves in the wild. "
            f"Zoologists explain this using the natural adaptations of "
            f"{config['niche'].lower()}. "
            f"And once you understand the real reason, "
            f"the animal kingdom suddenly looks a lot more fascinating."
        )

        title = (
            topic[0].upper() + topic[1:]
            if topic
            else topic
        )

    return {
        "title": title,
        "script": script,
    }


# ---------------------------------------------------------------------------
# Public Main API
# ---------------------------------------------------------------------------

def generate_script(
    channel_id: str,
    is_short: bool = False,
    forced_topic: str = None,
) -> dict:

    config = load_channel_config(
        channel_id
    )

    if forced_topic:
        topic = forced_topic
    else:
        topic = (
            get_daily_brain_topic(
                channel_id,
                config,
                is_short,
            )
            or get_trending_topic(
                channel_id,
                config,
            )
            or pick_next_topic(
                channel_id,
                config,
            )
        )

        mark_topic_used(
            channel_id,
            topic,
        )

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
            generated = generate_with_gemini(
                topic,
                config,
                length,
                language,
            )
        except Exception as e:
            print(
                f"WARNING: Gemini call failed ({e}); "
                "using template fallback."
            )
            generated = generate_with_template(
                topic,
                config,
                language,
            )
    else:
        generated = generate_with_template(
            topic,
            config,
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
    }


# ---------------------------------------------------------------------------
# Kids Animal Content
# ---------------------------------------------------------------------------

KIDS_CONTENT_TYPES = [
    "rhyme",
    "story",
    "learning",
]


def pick_content_type(
    config: dict,
) -> str:

    return random.choice(
        config.get("content_types")
        or KIDS_CONTENT_TYPES
    )


def _kids_seed_path(
    config: dict,
    content_type: str,
) -> Path:

    return (
        ROOT
        / config["topics_seed_files"][content_type]
    )


def _kids_dedupe_key(
    channel_id: str,
    content_type: str,
) -> str:

    return (
        f"{channel_id}__{content_type}"
    )


def _generate_more_kids_topics(
    config: dict,
    content_type: str,
    existing: list,
    count: int = 20,
) -> list:

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

        kind_description = {
            "rhyme": (
                "short original animal rhymes/poems themes "
                "for young children ages 2-6 (e.g. cute monkeys, jumping frogs)"
            ),
            "story": (
                "short animal moral story premises for young "
                "children, featuring forest animals or pets"
            ),
            "learning": (
                "simple animal learning topics for young children "
                "such as animal sounds, habitats, wild animals, and baby animals"
            ),
        }[content_type]

        prompt = f"""
You generate animal video theme ideas for a children's YouTube channel.

Channel:
{config['display_name']}

This batch is for:
{kind_description}

Already-used animal themes:

{existing_sample}

Do NOT repeat these or close variations.

Generate {count} brand new animal theme ideas.

Each theme must:
- Be one short line about animals.
- Be specific enough to build one kids video from.
- Be clearly different from existing themes.

No numbering.
No markdown.
No quotes.

Just one theme per line.
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
            "WARNING: kids animal topic auto-generation failed "
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
                f"Added {len(new_topics)} new animal "
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

    topic = random.choice(unused)

    mark_topic_used(
        dedupe_key,
        topic,
    )

    return topic


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
Write an ORIGINAL animal rhyme/poem for young children ages 2-6,
themed around:

{topic}

CRITICAL:
This must be a completely original composition about animals.
Do NOT reproduce or imitate copyrighted nursery rhymes.
Write it purely in Hindi using Devanagari script (or matching target language).
Use simple everyday words a toddler knows about animals.
Include a repeated animal sound or chorus.
Include at least one action kids can copy, such as: jump like a frog, walk like a bear.
The animal mascot {mascot_name} should lead it, mentioned warmly by name.
Keep the mood joyful and friendly.
"""

    elif content_type == "story":
        content_instructions = f"""
Write an ORIGINAL short animal moral story for young children ages 3-7,
themed around:

{topic}

The story should feature friendly animal characters led by {mascot_name}.
Teach a simple positive lesson like sharing, friendship, or helping others.
Show the lesson through the animal characters' actions.
Language style: Natural children's story style, simple and warm.
Nothing scary or sad.
"""

    else:
        content_instructions = f"""
Write an ORIGINAL short animal learning segment for young children ages 2-6,
teaching about:

{topic}

{mascot_name} the animal mascot should teach directly to the viewer.
Use a warm, encouraging, educational tone about animals and nature.
Use call-and-response phrasing.
"""

    prompt = f"""
You are writing an animal-themed script for a children's YouTube video.

Channel:
{config['display_name']}

Content type:
{content_type}

Language:
{language_name}

{content_instructions}

Respond in EXACTLY this format and nothing else:

TITLE: <a warm, simple title about animals in {language_name}, under 90 characters>

SCRIPT:
<spoken narration only>

No markdown.
No stage directions.
No visual cues.

Target script length:
approximately {words_target} words.
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


def generate_kids_template(
    topic: str,
    config: dict,
    content_type: str,
    mascot_name: str,
    language: str,
) -> dict:

    if content_type == "rhyme":
        script = (
            f"चलो सब मिलकर जानवरों की आवाज़ निकालें, {mascot_name} के साथ! "
            f"आज की प्यारी कविता है {topic} के बारे में। "
            f"छलांग लगाओ, खुश रहो! "
            f"बाय बाय दोस्तों!"
        )
        title = f"{mascot_name} और प्यारे जानवर की कविता"

    elif content_type == "story":
        script = (
            f"एक बार की बात है, जंगल में {mascot_name} रहता था। "
            f"उसने सीखा {topic} के बारे में एक बहुत प्यारी बात। "
            f"सब जानवर मिलकर खुश हुए। "
            f"The end!"
        )
        title = f"{mascot_name} की एनिमल स्टोरी"

    else:
        script = (
            f"नमस्ते दोस्तों! मैं हूँ {mascot_name}। "
            f"आज हम जानेंगे {topic} के बारे में। "
            f"Animals are our friends! "
            f"Great job, दोस्तों!"
        )
        title = f"{mascot_name} के साथ सीखो: {topic}"

    return {
        "title": title,
        "script": script,
    }


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

    mascot = config["mascot_map"][content_type]
    mascot_name = config["mascot_names"][mascot]

    if (
        content_type == "rhyme"
        and config.get("rhymes_hindi_only", True)
    ):
        language = "hi"
    else:
        language = pick_language(config)

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
    parser.add_argument("channel_id")
    parser.add_argument("--short", action="store_true")
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
        Path(args.out).write_text(output)

    print(output)
