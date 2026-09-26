"""Batch topic generation for Animals & Wildlife."""

from src.agents.orchestrator import run_json_batch


def find_topics(
    channel_id: str,
    config: dict,
    trend_signals: list[str],
    used_topics: list[str],
    count: int = 30,
) -> list[dict]:

    signals = (
        "\n".join(
            f"- {x}"
            for x in trend_signals[:40]
        )
        or "(no live signals)"
    )

    used = (
        "\n".join(
            f"- {x}"
            for x in used_topics[-150:]
        )
        or "(none)"
    )

    result = run_json_batch(
        "Animal & Wildlife Topic Hunter",
        "Find original, high-retention animal and wildlife facts without copying source titles.",
        f"""
Generate exactly {count} candidate topics.

IMPORTANT:
ALL topics MUST be about animals or wildlife.

Allowed subjects include:
- Strange animal abilities
- Animal senses
- Animal behavior
- Survival skills
- Unusual adaptations
- Record-breaking animals
- Dangerous animals
- Cute but surprising animal facts
- Deep-sea animals
- Rare animals
- Animal intelligence
- Predators and prey
- Unexpected animal relationships
- Animal communication
- Extreme animal survival
- Incredible animal anatomy

DO NOT generate topics about:
- Space
- Planets
- Stars
- Galaxies
- Black holes
- Astronomy
- Physics
- The universe
- NASA
- Generic science unrelated to animals
- Human-only science topics

Each item must contain:
{{
  "topic": "specific animal or wildlife topic or question",
  "curiosity": 1,
  "novelty": 1,
  "visual": 1,
  "science_confidence": 1,
  "reason": "one short reason"
}}

Return JSON object:
{{"topics": [ ... ]}}

Use recent signals as inspiration, not as titles to copy.

Reject:
- Generic animal facts
- Boring school-style topics
- Listicles
- Unsupported claims
- Fake or exaggerated claims
- Recycled ideas
- Misleading topics

Use real, well-established animal and wildlife facts.

Make the topics surprising, visual, clickable, and suitable for YouTube Shorts.

RECENT SIGNALS:
{signals}

ALREADY USED CHANNEL TOPICS:
{used}
""",
        f"""
CHANNEL: {config['display_name']}
NICHE: Animals & wildlife facts
TONE: {config['tone']}
""",
        max_output_tokens=4000,
    )

    topics = (
        result.get("topics", [])
        if isinstance(result, dict)
        else []
    )

    cleaned = []
    seen = set()

    for item in topics:
        if not isinstance(item, dict):
            continue

        topic = str(
            item.get("topic", "")
        ).strip()

        if not topic or len(topic) > 220:
            continue

        key = topic.casefold()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(item)

    return cleaned