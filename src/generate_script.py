
"""
generate_script.py

Generates YouTube scripts for the existing Autotube pipeline.

Includes:
- Winner Hunter, Daily Brain and Trend Scout.
- Topic deduplication and unused-topic history.
- Automatic topic generation when seeds run out.
- Same-day longform-to-Short recap support.
- Stronger hooks and tighter pacing for Shorts.
- Kids content compatibility helpers.
- Existing generate_script() return fields and CLI.
"""

import os
import json
import random
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.trend_scout import get_trending_topic
from src.daily_brain_topics import get_daily_brain_topic

try:
    from src.winner_hunter import build_winner_context
    WINNER_HUNTER_AVAILABLE = True
except ImportError:
    WINNER_HUNTER_AVAILABLE = False

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
    return max(
        MIN_OUTPUT_TOKENS,
        int(words_target * TOKENS_PER_WORD) + TOKEN_SAFETY_MARGIN,
    )


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
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {
                str(item).strip()
                for item in data
                if str(item).strip()
            }
        return set()
    except (json.JSONDecodeError, OSError):
        return set()


def mark_topic_used(channel_id: str, topic: str) -> None:
    topic = str(topic).strip()
    if not topic:
        return

    used = get_used_topics(channel_id)
    used.add(topic)

    path = _used_topics_path(channel_id)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            sorted(used, key=str.casefold),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _normalize_topic(topic: str) -> str:
    topic = str(topic or "").casefold()
    topic = "".join(
        character if character.isalnum() else " "
        for character in topic
    )
    return re.sub(r"\s+", " ", topic).strip()


def _topic_similarity(topic_a: str, topic_b: str) -> float:
    a = _normalize_topic(topic_a)
    b = _normalize_topic(topic_b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence_score = SequenceMatcher(None, a, b).ratio()
    words_a = set(a.split())
    words_b = set(b.split())

    overlap_score = (
        len(words_a & words_b)
        / max(1, min(len(words_a), len(words_b)))
    )

    return max(sequence_score, overlap_score)


def _looks_like_duplicate_topic(topic: str, used_topics: set) -> bool:
    normalized = _normalize_topic(topic)

    if not normalized:
        return True

    for old_topic in used_topics:
        old_normalized = _normalize_topic(old_topic)

        if not old_normalized:
            continue

        if normalized == old_normalized:
            return True

        if _topic_similarity(normalized, old_normalized) >= 0.82:
            return True

    return False


def _fresh_candidate(
    channel_id: str,
    candidate: str | None,
    source: str,
) -> str | None:
    if not candidate:
        return None

    topic = str(candidate).strip().strip('"').strip("'").strip()

    if not topic:
        return None

    if "\n" in topic:
        topic = topic.splitlines()[0].strip()

    if _looks_like_duplicate_topic(topic, get_used_topics(channel_id)):
        print(f"{source} generated a duplicate topic; rejecting: {topic}")
        return None

    print(f"{source} selected fresh topic: {topic}")
    return topic


def _todays_longform_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_todays_longform.json"


def save_todays_longform_topic(channel_id: str, topic: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    _todays_longform_path(channel_id).write_text(
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
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(topic).strip()


def _generate_with_token_budget(
    client,
    prompt: str,
    max_output_tokens: int,
):
    try:
        from google.genai import types

        generation_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens
        )
        chat = client.chats.create(
            model=GEMINI_MODEL,
            config=generation_config,
        )
        response = chat.send_message(prompt)

    except Exception as first_error:
        print(
            "WARNING: Gemini configured call failed "
            f"({first_error}); retrying without explicit config."
        )
        chat = client.chats.create(model=GEMINI_MODEL)
        response = chat.send_message(prompt)

    try:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "MAX_TOKENS" in str(finish_reason).upper():
            print(
                "WARNING: Gemini response hit "
                f"max_output_tokens={max_output_tokens}."
            )
    except Exception:
        pass

    return response


def _parse_topic_lines(response_text: str) -> list[str]:
    topics = []

    for line in (response_text or "").splitlines():
        topic = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", line).strip()
        topic = topic.strip('"').strip("'").strip()

        if topic:
            topics.append(topic)

    return topics


def _generate_more_topics(
    config: dict,
    existing: list,
    count: int = 20,
) -> list:
    if not (GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY")):
        return []

    try:
        client = genai_client.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        existing_sample = "\n".join(
            f"- {topic}" for topic in existing[-100:]
        )

        prompt = f"""
Generate {count} fresh topics for a faceless YouTube channel.

CHANNEL: {config.get('display_name', 'Animal Discovery')}
NICHE: Animals & wildlife facts
TONE: {config.get('tone', 'curious, surprising and conversational')}

Already used or already in the seed list:
{existing_sample}

Every topic must concern real animals or wildlife. Explore different
species, senses, behaviors, survival adaptations, intelligence, predators,
prey, rare animals, deep-sea animals, and unusual animal relationships.

Do not generate space, astronomy, NASA, unrelated science, or fictional
animal abilities. Avoid duplicate subjects and close rewordings.

Return one topic per line, without numbering or explanations.
"""

        response = _generate_with_token_budget(client, prompt, 1200)
        channel_id = config.get("channel_id", "")
        used_history = set(existing)

        if channel_id:
            used_history.update(get_used_topics(channel_id))

        result = []

        for topic in _parse_topic_lines(response.text):
            if _looks_like_duplicate_topic(topic, used_history):
                continue
            if _looks_like_duplicate_topic(topic, set(result)):
                continue
            result.append(topic)

        return result

    except Exception as e:
        print(f"WARNING: topic auto-generation failed ({e}).")
        return []


def _generate_winner_inspired_topic(
    channel_id: str,
    config: dict,
    is_short: bool,
) -> str | None:
    if not WINNER_HUNTER_AVAILABLE:
        print("Winner Hunter unavailable.")
        return None

    if not (GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY")):
        print("Gemini unavailable; Winner Hunter cannot generate candidates.")
        return None

    try:
        winner_context = build_winner_context(channel_id)
        if not winner_context:
            print("Winner Hunter has no performance context.")
            return None
    except Exception as e:
        print(f"WARNING: Winner Hunter context failed ({e}).")
        return None

    used_topics = get_used_topics(channel_id)
    used_sample = "\n".join(
        f"- {topic}"
        for topic in sorted(used_topics, key=str.casefold)[-120:]
    )

    content_length = "a YouTube Short" if is_short else "a YouTube video"

    prompt = f"""
You are the topic strategist for a faceless animal and wildlife channel.
Generate exactly 12 distinct candidate topics for {content_length}.

CHANNEL PERFORMANCE DATA:
{winner_context}

PREVIOUSLY USED TOPICS. Do not repeat these or their underlying subjects:
{used_sample}

Reuse successful CONTENT PATTERNS, not the same subjects. Find different
animals, species, behaviors, abilities, adaptations, and survival methods.

Topics must be specific, fact-checkable, and curiosity-driven.
Never invent species, behavior, abilities, statistics, or discoveries.
No astronomy, space, NASA, or unrelated science.

Return exactly 12 topics, one per line, without numbering or commentary.
"""

    try:
        client = genai_client.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        response = _generate_with_token_budget(client, prompt, 1200)

        candidates = []
        comparison_set = set(used_topics)

        for candidate in _parse_topic_lines(response.text):
            if _looks_like_duplicate_topic(candidate, comparison_set):
                print(f"Winner Hunter duplicate rejected: {candidate}")
                continue

            candidates.append(candidate)
            comparison_set.add(candidate)

        if not candidates:
            print("Winner Hunter returned no fresh candidates.")
            return None

        selected = random.choice(candidates)
        print(
            f"Winner Hunter selected from {len(candidates)} fresh "
            f"candidates: {selected}"
        )
        return selected

    except Exception as e:
        print(f"WARNING: Winner Hunter generation failed ({e}).")
        return None


def pick_next_topic(channel_id: str, config: dict) -> str:
    topics_file = ROOT / config["topics_seed_file"]
    all_topics = [
        line.strip()
        for line in topics_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    used = get_used_topics(channel_id)
    unused = [
        topic for topic in all_topics
        if not _looks_like_duplicate_topic(topic, used)
    ]

    if not unused:
        generation_config = dict(config)
        generation_config["channel_id"] = channel_id

        new_topics = _generate_more_topics(
            generation_config,
            list(dict.fromkeys(all_topics + sorted(used, key=str.casefold))),
        )

        if new_topics:
            with topics_file.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_topics) + "\n")

            unused = [
                topic for topic in new_topics
                if not _looks_like_duplicate_topic(topic, used)
            ]

    if not unused:
        raise RuntimeError(
            f"No fresh topics available for channel '{channel_id}'. "
            "Add seed topics or enable Gemini topic generation."
        )

    topic = random.choice(unused)
    mark_topic_used(channel_id, topic)
    return topic


def pick_language(config: dict) -> str:
    return random.choice(config.get("languages") or ["en"])


def _voice_for_language(config: dict, language: str) -> str:
    voices = config.get("voices") or {}
    voice = voices.get(language) or config.get("voice")

    if not voice:
        raise ValueError(
            f"No TTS voice configured for language '{language}'."
        )

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
        "title": (
            title_match.group(1).strip()
            if title_match
            else fallback_topic
        ),
        "script": _clean_script_text(
            script_match.group(1).strip()
            if script_match
            else text
        ),
    }


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
        words_target = 105
    elif length_seconds <= 55:
        words_target = 125
    else:
        words_target = round(length_seconds / 60 * 140)

    language_name = LANGUAGE_NAMES.get(language, language)
    is_short = length_seconds <= 60

    if is_short:
        pacing_rules = """
SHORTS RETENTION RULES:
- The first sentence must create immediate curiosity in the first 1-2 seconds.
- Start with a concrete, surprising fact or consequence about the animal.
- Never begin with "Did you know", "Have you ever wondered", "Today we're
  going to talk about", "Imagine", a greeting, or the topic name alone.
- Do not waste the opening on background or explanations.
- Make the first sentence understandable without any context.
- Every sentence must introduce a new fact, escalation, contrast, or payoff.
- Remove filler, repeated facts, and sentences that merely connect ideas.
- Build one clear curiosity loop and resolve it before the ending.
- The payoff must be specific and supported by a real fact.
- Do not use fake cliffhangers or promise a reveal that never arrives.
- Do not add a slow introduction or a generic conclusion.
- Aim for roughly 90-115 words for a 45-second Short.
"""
    else:
        pacing_rules = """
LONG-FORM RETENTION RULES:
- Open with a concrete, surprising fact or consequence.
- Avoid greetings, generic introductions, and repeating the title.
- Establish the central question quickly.
- Introduce new information in every paragraph.
- Build curiosity through specific facts, contrasts, and escalating reveals.
- Resolve the main question with a meaningful factual payoff.
"""

    prompt = f"""
You are the lead writer for a faceless YouTube channel called
"{config['display_name']}".

NICHE: Animals and wildlife facts.
TONE: Curious, surprising, energetic, punchy, conversational.
TOPIC: {topic}

Write a high-retention voiceover about this exact topic.

CONTENT RULES:
- The entire script must be about animals or wildlife.
- Use real, well-established facts.
- Never invent statistics, discoveries, quotes, experiments, behaviors,
  or abilities.
- Never exaggerate abilities or present speculation as fact.
- Do not write about space, planets, stars, galaxies, black holes,
  astronomy, NASA, the universe, or unrelated physics.
- Stay focused on the specific animal and topic.
- Do not invent a "shocking" fact just to make the script exciting.

{pacing_rules}

STYLE:
- Natural spoken language suitable for text-to-speech.
- Prefer short, clear sentences.
- Vary sentence length without making sentences complicated.
- No academic language, filler, generic listicles, motivational lessons,
  emojis, stage directions, visual labels, or markdown.
- Do not begin by simply repeating the topic.
- Do not mention AI.
- Do not say subscribe, like, or follow.
- Do not use unsupported superlatives such as "the most powerful ever".
- Make the title specific, intriguing, and accurate. No misleading clickbait.

LANGUAGE: Write entirely in {language_name}.
If Hindi is requested, use Devanagari, not Romanized Hindi.
LENGTH: approximately {words_target} words. Do not pad with repetition.

Before answering, silently check:
1. Does the first sentence make someone want to keep watching?
2. Is the opening specific to this topic rather than interchangeable?
3. Does every sentence add new information?
4. Is the payoff factual and actually delivered?
5. Is the script appropriate for the requested length?

OUTPUT EXACTLY:
TITLE: <punchy YouTube title under 90 characters>
SCRIPT:
<spoken narration only>
"""

    response = _generate_with_token_budget(
        client,
        prompt,
        _max_output_tokens_for(words_target),
    )

    result = _parse_titled_response(response.text, topic)
    result["title"] = result["title"].strip().strip('"').strip("'")

    word_count = len(result["script"].split())
    if is_short and word_count > 145:
        print(
            f"WARNING: Short script has {word_count} words; "
            "it may be too long for the target duration."
        )

    if not result["script"]:
        raise ValueError("Gemini returned an empty script.")

    return result


def generate_with_template(
    topic: str,
    config: dict,
    length_seconds: int,
    language: str,
) -> dict:
    """
    Minimal fallback used when Gemini is unavailable.

    This avoids pretending to know specific animal facts that were not
    generated or verified. The fallback is intentionally short.
    """
    if language == "hi":
        script = (
            f"इस विषय में जानवरों की दुनिया का एक दिलचस्प पहलू छिपा है: "
            f"{topic}। किसी भी दावे को सच मानने से पहले उसके प्रमाण को "
            "समझना ज़रूरी है। प्रकृति में हर व्यवहार के पीछे एक कारण "
            "हो सकता है, और सही जानकारी हमें उस कारण को समझने में मदद "
            "करती है।"
        )
    else:
        script = (
            f"What makes this topic worth investigating is the animal "
            f"behavior behind it: {topic}. The important question is "
            "what the animal actually does, and why. Real wildlife can "
            "be surprising, but the explanation matters just as much as "
            "the surprise. The most interesting detail is the one that "
            "the evidence can support, not a claim that merely sounds "
            "impressive."
        )

    return {
        "title": topic.strip(),
        "script": script,
    }


def generate_script(
    channel_id: str,
    is_short: bool = False,
    forced_topic: str | None = None,
) -> dict:
    """
    Topic priority:
    1. Explicit forced topic.
    2. One eligible same-day longform recap for a Short.
    3. Winner Hunter.
    4. Daily Brain.
    5. Trend Scout.
    6. An unused seed topic or newly generated topic.
    """
    config = load_channel_config(channel_id)

    length = (
        config["short_length_seconds"]
        if is_short
        else config["video_length_seconds"]
    )
    language = pick_language(config)

    topic = forced_topic.strip() if forced_topic else None
    recap_topic = False

    if not topic and is_short:
        topic = get_recap_topic_for_short(channel_id)
        recap_topic = bool(topic)

        if topic:
            print(f"Using today's longform topic for recap Short: {topic}")

    if not topic:
        try:
            candidate = _generate_winner_inspired_topic(
                channel_id, config, is_short
            )
            topic = _fresh_candidate(
                channel_id, candidate, "Winner Hunter"
            )
        except Exception as e:
            print(f"WARNING: Winner Hunter selection failed ({e}).")

    if not topic:
        try:
            candidate = get_daily_brain_topic(
                channel_id, config, is_short
            )
            topic = _fresh_candidate(
                channel_id, candidate, "Daily Brain"
            )
        except Exception as e:
            print(f"WARNING: Daily Brain selection failed ({e}).")

    if not topic:
        try:
            candidate = get_trending_topic(channel_id, config)
            topic = _fresh_candidate(
                channel_id, candidate, "Trend Scout"
            )
        except Exception as e:
            print(f"WARNING: Trend Scout selection failed ({e}).")

    if not topic:
        topic = pick_next_topic(channel_id, config)

    if not forced_topic and not recap_topic:
        mark_topic_used(channel_id, topic)

    if not is_short and not forced_topic:
        save_todays_longform_topic(channel_id, topic)

    generated = None

    if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
        try:
            generated = generate_with_gemini(
                topic, config, length, language
            )
        except Exception as e:
            print(
                f"WARNING: Gemini script generation failed ({e}); "
                "using template fallback."
            )

    if not generated:
        generated = generate_with_template(
            topic, config, length, language
        )

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

    path_value = (
        files.get(content_type)
        or config.get(f"{content_type}_topics_seed_file")
        or config.get("kids_topics_seed_file")
    )

    if not path_value:
        raise KeyError(
            f"No kids topic seed file configured for '{content_type}'."
        )

    return ROOT / path_value


def _kids_dedupe_key(channel_id: str, content_type: str) -> str:
    return f"{channel_id}_kids_{content_type}"


def pick_content_type(config: dict) -> str:
    choices = (
        config.get("kids_content_types")
        or config.get("content_types")
        or ["lesson", "story", "rhyme"]
    )

    if isinstance(choices, dict):
        choices = list(choices.keys())

    return random.choice(list(choices)) if choices else "lesson"


def _generate_more_kids_topics(
    config: dict,
    content_type: str,
    existing: list,
    count: int = 20,
) -> list:
    if not (GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY")):
        return []

    try:
        client = genai_client.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        existing_sample = "\n".join(
            f"- {topic}" for topic in existing[-60:]
        )

        prompt = f"""
Generate {count} original children's YouTube topic ideas.

Channel: {config.get('display_name', 'Kids channel')}
Content type: {content_type}

Already used or already in the seed list:
{existing_sample}

Each topic must fit the requested content type and be suitable for
young children. Do not copy existing topics or copyrighted song/story
titles. Return one topic per line without numbering or commentary.
"""

        response = _generate_with_token_budget(client, prompt, 900)
        result = []

        for topic in _parse_topic_lines(response.text):
            if _looks_like_duplicate_topic(
                topic, set(existing) | set(result)
            ):
                continue
            result.append(topic)

        return result

    except Exception as e:
        print(f"WARNING: kids topic auto-generation failed ({e}).")
        return []


def pick_next_kids_topic(
    channel_id: str,
    config: dict,
    content_type: str,
) -> str:
    topics_file = _kids_seed_path(config, content_type)
    dedupe_key = _kids_dedupe_key(channel_id, content_type)

    all_topics = [
        line.strip()
        for line in topics_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    used = get_used_topics(dedupe_key)

    unused = [
        topic for topic in all_topics
        if not _looks_like_duplicate_topic(topic, used)
    ]

    if not unused:
        new_topics = _generate_more_kids_topics(
            config,
            content_type,
            list(dict.fromkeys(
                all_topics + sorted(used, key=str.casefold)
            )),
        )

        if new_topics:
            with topics_file.open("a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_topics) + "\n")

            unused = [
                topic for topic in new_topics
                if not _looks_like_duplicate_topic(topic, used)
            ]

    if not unused:
        raise RuntimeError(
            f"No fresh topics available for kids content type "
            f"'{content_type}'. Add seed topics or enable Gemini "
            "topic generation; refusing to recycle topics."
        )

    topic = random.choice(unused)
    mark_topic_used(dedupe_key, topic)
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

    words_target = int(length_seconds * 2.2)
    language_name = LANGUAGE_NAMES.get(language, language)

    if content_type == "rhyme":
        content_instructions = f"""
Write an ORIGINAL Hindi rhyme/poem for children ages 2-6 about:
{topic}

Do not reproduce or closely imitate an existing nursery rhyme, song,
or poem. Use Devanagari, simple words, clear rhythm, a short refrain,
and one easy action children can copy. {mascot_name} should lead it.
Keep it joyful and gentle.
"""
    elif content_type == "story":
        content_instructions = f"""
Write an ORIGINAL short moral story for children ages 3-7 about:
{topic}

Feature {mascot_name}. Teach one simple positive lesson through the
story. Use short sentences and gentle conflict. Nothing scary, violent,
or sad. Natural Hinglish code-mixing is allowed.
"""
    else:
        content_instructions = f"""
Write an ORIGINAL short learning segment for children ages 2-6 teaching:
{topic}

{mascot_name} teaches directly to the viewer using warm call-and-response
phrasing. Use simple bilingual teaching with Hindi and everyday English.
"""

    prompt = f"""
You are writing a children's YouTube video script.
Channel: {config['display_name']}
Content type: {content_type}
Language: {language_name}

{content_instructions}

Start immediately with an engaging line. Avoid a long introduction.
Use short spoken sentences and make each line useful or fun.
Keep the content age-appropriate and original.

Respond exactly:
TITLE: <warm title under 90 characters>
SCRIPT:
<spoken narration only>

No markdown, stage directions, visual cues, or commentary.
Target approximately {words_target} words. End warmly.
"""

    response = _generate_with_token_budget(
        client, prompt, _max_output_tokens_for(words_target)
    )

    return _parse_titled_response(response.text, topic)


def generate_kids_template(
    topic: str,
    config: dict,
    content_type: str,
    mascot_name: str,
    language: str,
) -> dict:
    if content_type == "rhyme":
        return {
            "title": f"{mascot_name} की मस्ती भरी कविता",
            "script": (
                f"चलो सब मिलकर गाएं, {mascot_name} के साथ। "
                f"आज की कविता है {topic} के बारे में। "
                "ताली बजाओ, संग गाओ, मज़ा करो, हाँ! "
                "फिर मिलेंगे, बाय बाय!"
            ),
        }

    if content_type == "story":
        return {
            "title": f"{mascot_name} की एक प्यारी कहानी",
            "script": (
                f"एक बार की बात है, {mascot_name} नाम का एक प्यारा दोस्त था। "
                f"एक दिन उसने {topic} के बारे में कुछ नया सीखा। "
                "उसने अपने दोस्तों के साथ मिलकर इस सीख को समझा। "
                "अंत में सब दोस्त खुश हुए। The end!"
            ),
        }

    return {
        "title": f"{mascot_name} के साथ सीखो: {topic}",
        "script": (
            f"नमस्ते दोस्तों! मैं हूँ {mascot_name}। "
            f"आज हम सीखेंगे {topic}। "
            "बोलो मेरे साथ! बहुत बढ़िया! "
            "Great job, दोस्तों! फिर मिलेंगे अगली सीख के साथ!"
        ),
    }


def generate_kids_script(
    channel_id: str,
    is_short: bool = False,
) -> dict:
    config = load_channel_config(channel_id)
    content_type = pick_content_type(config)

    topic = pick_next_kids_topic(
        channel_id, config, content_type
    )

    mascot_map = config.get("mascot_map") or {}
    mascot_names = config.get("mascot_names") or {}

    mascot = mascot_map.get(
        content_type,
        next(iter(mascot_names), "mascot"),
    )
    mascot_name = mascot_names.get(mascot, str(mascot))

    language = (
        "hi"
        if content_type == "rhyme"
        and config.get("rhymes_hindi_only", True)
        else pick_language(config)
    )

    length = (
        config["short_length_seconds"]
        if is_short
        else config["video_length_seconds"]
    )

    try:
        if GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
            generated = generate_kids_script_with_gemini(
                topic,
                config,
                content_type,
                mascot_name,
                length,
                language,
            )
        else:
            generated = generate_kids_template(
                topic, config, content_type, mascot_name, language
            )
    except Exception as e:
        print(
            f"WARNING: kids Gemini generation failed ({e}); "
            "using template fallback."
        )
        generated = generate_kids_template(
            topic, config, content_type, mascot_name, language
        )

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
        Path(args.out).write_text(output, encoding="utf-8")

    print(output)
