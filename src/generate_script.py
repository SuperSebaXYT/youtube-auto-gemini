"""
generate_script.py

Generates scripts for the Autotube channels using Google's current
google-genai SDK.

IMPORTANT:
- Uses client.chats.create() + chat.send_message()
- Does NOT use client.models.generate_content() for text generation.
- Integrates the separate winner_hunter.py module.
- Winner Hunter learns from real YouTube Analytics performance.
- Falls back safely if Winner Hunter / Analytics is unavailable.
- Falls back to template scripts if Gemini is unavailable.
"""

import os
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher

import yaml

from src.trend_scout import get_trending_topic
from src.daily_brain_topics import get_daily_brain_topic


# ---------------------------------------------------------------------------
# Winner Hunter
# ---------------------------------------------------------------------------

try:
    from src.winner_hunter import build_winner_context
except Exception:
    build_winner_context = None


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

    with open(path, "r", encoding="utf-8") as f:
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
            return set(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return set()

    return set()


def mark_topic_used(channel_id: str, topic: str) -> None:
    used = get_used_topics(channel_id)
    used.add(topic)

    _used_topics_path(channel_id).write_text(
        json.dumps(
            sorted(used),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Topic similarity
# ---------------------------------------------------------------------------

def _normalize_topic(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _topics_are_similar(
    first: str,
    second: str,
) -> bool:
    """
    Prevents Winner Hunter from selecting exact duplicates
    or obvious rewrites of existing topics.
    """

    first_normalized = _normalize_topic(first)
    second_normalized = _normalize_topic(second)

    if not first_normalized or not second_normalized:
        return False

    if first_normalized == second_normalized:
        return True

    similarity = SequenceMatcher(
        None,
        first_normalized,
        second_normalized,
    ).ratio()

    if similarity >= 0.82:
        return True

    first_words = set(first_normalized.split())
    second_words = set(second_normalized.split())

    if not first_words or not second_words:
        return False

    overlap = (
        len(first_words.intersection(second_words))
        / min(len(first_words), len(second_words))
    )

    return overlap >= 0.75


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
    Uses:

        client.chats.create()
        chat.send_message()
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

    if (
        finish_reason
        and "MAX_TOKENS" in str(finish_reason).upper()
    ):
        print(
            "WARNING: Gemini response was truncated by "
            f"max_output_tokens={max_output_tokens}. "
            "The generated script may be shorter than the target."
        )

    return response


# ---------------------------------------------------------------------------
# Winner Hunter topic generation
# ---------------------------------------------------------------------------

def _generate_winner_hunter_topic(
    channel_id: str,
    config: dict,
) -> str | None:
    """
    Uses the separate winner_hunter.py module to obtain real channel
    performance patterns and asks Gemini to create a completely new topic.

    Winner Hunter itself performs the Analytics analysis.
    This function only turns those patterns into a new topic.
    """

    if not (
        GEMINI_AVAILABLE
        and os.environ.get("GEMINI_API_KEY")
        and build_winner_context
    ):
        return None

    try:
        winner_context = build_winner_context(
            channel_id
        )

        if not winner_context:
            return None

        used_topics = get_used_topics(
            channel_id
        )

        topics_file = ROOT / config["topics_seed_file"]

        seed_topics = []

        if topics_file.exists():
            seed_topics = [
                line.strip()
                for line in topics_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

        recent_used = list(used_topics)[-80:]

        existing_text = "\n".join(
            f"- {topic}"
            for topic in recent_used
        )

        if not existing_text:
            existing_text = "- None"

        client = genai_client.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        prompt = f"""
You are the growth topic strategist for a faceless YouTube channel
focused on animals and wildlife.

CHANNEL:
{config['display_name']}

NICHE:
{config['niche']}

TONE:
{config['tone']}

REAL WINNER HUNTER PERFORMANCE DATA:
{winner_context}

The data above comes from this channel's real YouTube Analytics.

Your job is to identify WHY certain content performs well and then
create completely NEW subjects using those successful patterns.

Do NOT copy existing videos.

Do NOT rewrite an existing title.

Do NOT make a near-duplicate of an existing topic.

Examples:

If extreme animal abilities perform well:
choose a NEW animal and a NEW ability.

If strange behavior performs well:
choose a NEW species and a DIFFERENT strange behavior.

If dangerous animals perform well:
choose a NEW animal and a DIFFERENT danger mechanism.

If survival adaptations perform well:
choose a NEW species and a DIFFERENT survival adaptation.

ONLY generate topics about:

- real animals
- wildlife
- animal behavior
- animal abilities
- animal adaptations
- animal survival
- animal intelligence
- predators and prey
- unusual animal biology
- surprising animal facts

DO NOT generate:

- space
- planets
- technology
- politics
- humans as the main subject
- generic science
- motivation
- fictional animals
- fake discoveries

ALREADY USED TOPICS:

{existing_text}

Generate 12 possible NEW topics.

Every topic must:

- focus on a specific animal or animal group
- have a strong curiosity angle
- contain a real biological subject
- be specific enough for a complete video
- be meaningfully different from previous topics
- avoid unsupported clickbait

No numbering.
No bullets.
No quotes.

One topic per line.
"""

        response = _generate_with_token_budget(
            client,
            prompt,
            max_output_tokens=900,
        )

        candidates = []

        for line in response.text.splitlines():

            candidate = re.sub(
                r"^[\d\.\-\)\s]+",
                "",
                line,
            ).strip()

            candidate = candidate.strip(
                "\"'"
            )

            if candidate:
                candidates.append(candidate)

        all_existing = (
            list(used_topics)
            + seed_topics
        )

        for candidate in candidates:

            if any(
                _topics_are_similar(
                    candidate,
                    old_topic,
                )
                for old_topic in all_existing
            ):
                continue

            mark_topic_used(
                channel_id,
                candidate,
            )

            print(
                "Winner Hunter selected NEW topic: "
                f"{candidate}"
            )

            return candidate

        print(
            "Winner Hunter generated candidates, "
            "but all candidates were duplicates or too similar."
        )

        return None

    except Exception as e:

        print(
            "WARNING: Winner Hunter topic generation failed "
            f"({e}); continuing with normal topic selection."
        )

        return None


# ---------------------------------------------------------------------------
# Long-form -> Short recap state
# ---------------------------------------------------------------------------

def _todays_longform_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_todays_longform.json"


def save_todays_longform_topic(
    channel_id: str,
    topic: str,
) -> None:

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    _todays_longform_path(
        channel_id
    ).write_text(
        json.dumps(
            {
                "date": today,
                "topic": topic,
                "used_for_short": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_recap_topic_for_short(
    channel_id: str,
) -> str | None:

    path = _todays_longform_path(
        channel_id
    )

    if not path.exists():
        return None

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

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
        ),
        encoding="utf-8",
    )

    return topic


# ---------------------------------------------------------------------------
# Normal topic generation
# ---------------------------------------------------------------------------

def _generate_more_topics(
    config: dict,
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

        prompt = f"""
You generate topic ideas for a faceless YouTube channel focused on
animals and wildlife facts.

Channel:
{config['display_name']}

Niche:
{config['niche']}

Tone:
{config['tone']}

Here are topics already covered.
Do NOT repeat these or close variations:

{existing_sample}

Generate {count} brand new topic ideas for this channel focusing on
animals, wildlife, animal behavior, biology, and ecology.

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

        return [
            line
            for line in lines
            if line
            and line not in existing
        ]

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
        for line in topics_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    used = get_used_topics(
        channel_id
    )

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

            with topics_file.open(
                "a",
                encoding="utf-8",
            ) as f:
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

    topic = random.choice(
        unused
    )

    mark_topic_used(
        channel_id,
        topic,
    )

    return topic


# ---------------------------------------------------------------------------
# Language / voice
# ---------------------------------------------------------------------------

def pick_language(
    config: dict,
) -> str:

    languages = (
        config.get("languages")
        or ["en"]
    )

    return random.choice(
        languages
    )


def _voice_for_language(
    config: dict,
    language: str,
) -> str:

    voices = (
        config.get("voices")
        or {}
    )

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

def _clean_script_text(
    text: str,
) -> str:

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
        "script": _clean_script_text(
            script_raw
        ),
    }


# ---------------------------------------------------------------------------
# Animals & Wildlife Facts Gemini generation
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
        words_target = round(
            length_seconds / 60 * 140
        )

    language_name = LANGUAGE_NAMES.get(
        language,
        language,
    )

    prompt = f"""
You are the lead wildlife and biology writer for a premium faceless
YouTube channel called "{config['display_name']}".

CHANNEL NICHE:
{config['niche']}

CHANNEL TONE:
{config['tone']}

TOPIC:
{topic}

Write a highly engaging wildlife, animal behavior, and biology voiceover.

The viewer should feel:

"I had no idea animals could do that."

IMPORTANT:

This is NOT a generic motivational video.
This is NOT a list of random facts.
This is NOT an introduction to basic biology.

The entire script must be specifically about the supplied topic.

FACTUAL STANDARD:

- Use established biological and ecological knowledge whenever possible.
- Do not invent statistics, discoveries, quotes, experiments, or findings.
- Do not present speculation as fact.
- If the topic involves a hypothesis or unusual animal behavior, clearly signal
  that researchers have observed it or that it is studied by biologists.
- Prefer concrete biological explanations over vague descriptions.
- If a precise number is uncertain or unnecessary, don't invent one.
- Avoid sensational claims that contradict established science.
- Do not use physically impossible explanations just to make the story
  sound dramatic.

RETENTION STRUCTURE:

1. HOOK

Start with the most surprising consequence, question, or image about the animal.

Do NOT simply repeat the topic.

2. SETUP

Give just enough context for the viewer to understand the animal and its environment.

3. ESCALATION

Explain the biological process or behavior step by step.

Each sentence should make the situation more interesting.

4. PAYOFF

Reveal the strangest, most surprising, or least-known consequence of this animal trait.

5. FINAL LINE

End on a memorable biological thought connected directly to the topic.

Do NOT say:
subscribe
like
follow
in conclusion

WRITING RULES:

- Conversational spoken language.
- Short sentences mixed with occasional longer sentences.
- No academic-paper language.
- No unnecessary definitions.
- No filler.
- No motivational life lessons.
- No generic phrases such as:
  "Here's something most people don't know"
  "It sounds simple"
  "This reveals a lot about nature"
  "Once you understand"
  "you'll start noticing"
  "changes how you see the world"
  "in the wild"
  "nature is full of mysteries"
- Do not begin by repeating the topic.
- Do not use rhetorical filler every few sentences.
- Every sentence must either create curiosity, explain something,
  or deliver a payoff.
- Do not use emojis.
- Do not use stage directions.
- Do not use visual notes.
- Do not use markdown.
- Do not mention that you are an AI.
- The narration must sound natural when read by text-to-speech.

VISUAL THINKING:

Write sentences that naturally correspond to visual moments.

For example, if explaining an animal hunting or defense mechanism, the narration might naturally
move through:

habitat -> stealth -> ambush -> pursuit -> escape.

Do not literally write those visual labels into the script.

LANGUAGE:

Write entirely in {language_name}.

If the requested language is Hindi, use natural modern Hindi in Devanagari,
not Romanized Hindi.

LENGTH:

Approximately {words_target} words.

Do not pad the script merely to reach the word count.

A slightly shorter excellent script is better than a longer repetitive one.

OUTPUT FORMAT:

Respond in EXACTLY this format and nothing else:

TITLE: <punchy YouTube title under 90 characters>

SCRIPT:
<spoken narration only>
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
# Emergency template fallback
# ---------------------------------------------------------------------------

def generate_with_template(
    topic: str,
    config: dict,
    language: str,
) -> dict:

    if language == "hi":

        script = (
            f"क्या आपने कभी सोचा है कि {topic} के पीछे असल में क्या होता है? "
            f"यह सवाल जितना आसान लगता है, इसकी असली कहानी उतनी ही दिलचस्प है। "
            f"वैज्ञानिक और जीव-विज्ञान विशेषज्ञ इस व्यवहार को समझने के लिए "
            f"प्राकृतिक अनुकूलन का अध्ययन करते हैं। और सबसे दिलचस्प बात यह है "
            f"कि इसका जवाब हमारी सोच से कहीं ज्यादा जटिल है।"
        )

        title = f"{topic}"

    else:

        script = (
            f"{topic}. "
            f"The surprising part is what happens next. "
            f"Biologists can explain this using the remarkable adaptations behind "
            f"{config['niche'].lower()}. "
            f"And once you understand the real reason, "
            f"the natural world suddenly looks a little stranger."
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
# Public API
# ---------------------------------------------------------------------------

def generate_script(
    channel_id: str,
    is_short: bool = False,
    forced_topic: str = None,
) -> dict:

    config = load_channel_config(
        channel_id
    )

    topic = None
    topic_already_marked_used = False

    # -------------------------------------------------------
    # 1. Forced topic
    # -------------------------------------------------------

    if forced_topic:

        topic = forced_topic

    else:

        # ---------------------------------------------------
        # 2. Winner Hunter
        # ---------------------------------------------------

        winner_topic = _generate_winner_hunter_topic(
            channel_id,
            config,
        )

        if winner_topic:

            topic = winner_topic
            topic_already_marked_used = True

        # ---------------------------------------------------
        # 3. Daily Brain
        # ---------------------------------------------------

        if not topic:

            try:

                topic = get_daily_brain_topic(
                    channel_id,
                    config,
                    is_short,
                )

            except Exception as e:

                print(
                    "WARNING: Daily Brain topic selection failed "
                    f"({e}); continuing."
                )

                topic = None

        # ---------------------------------------------------
        # 4. Trend Scout
        # ---------------------------------------------------

        if not topic:

            try:

                topic = get_trending_topic(
                    channel_id,
                    config,
                )

            except Exception as e:

                print(
                    "WARNING: Trend Scout topic selection failed "
                    f"({e}); continuing."
                )

                topic = None

        # ---------------------------------------------------
        # 5. Static topic pool
        # ---------------------------------------------------

        if not topic:

            topic = pick_next_topic(
                channel_id,
                config,
            )

            topic_already_marked_used = True

        # Daily Brain / Trend Scout topics are marked here.
        if (
            topic
            and not topic_already_marked_used
        ):

            mark_topic_used(
                channel_id,
                topic,
            )

    if not topic:

        raise RuntimeError(
            f"Could not select a topic for channel "
            f"'{channel_id}'."
        )

    language = pick_language(
        config
    )

    length = (
        config["short_length_seconds"]
        if is_short
        else config["video_length_seconds"]
    )

    # -------------------------------------------------------
    # Script generation
    # -------------------------------------------------------

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
# Kids content
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


# ---------------------------------------------------------------------------
# Kids topic generation
# ---------------------------------------------------------------------------

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
                "short original Hindi rhyme/poem themes "
                "for young children ages 2-6"
            ),
            "story": (
                "short moral story premises for young "
                "children, gentle and positive"
            ),
            "learning": (
                "simple learning topics for young children "
                "such as letters, numbers, colors, shapes, "
                "and categories"
            ),
        }[content_type]

        prompt = f"""
You generate video theme ideas for a children's YouTube channel.

Channel:
{config['display_name']}

This batch is for:
{kind_description}

Already-used themes:

{existing_sample}

Do NOT repeat these or close variations.

Generate {count} brand new theme ideas.

Each theme must:
- Be one short line.
- Be specific enough to build one video from.
- Be clearly different from the existing themes.

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

        return [
            line
            for line in lines
            if line
            and line not in existing
        ]

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
        for line in topics_file.read_text(
            encoding="utf-8"
        ).splitlines()
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

            with topics_file.open(
                "a",
                encoding="utf-8",
            ) as f:
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
            output,
            encoding="utf-8",
        )

    print(output)
