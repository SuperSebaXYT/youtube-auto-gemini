"""
trend_scout.py

Current-signal discovery for Animals & Wildlife.

Strategy:

YouTube -> optional Reddit -> Gemini synthesis

Reddit is treated as an optional signal only.

If Reddit returns 403/429/network errors, the system continues without it.

If no trend source works, the caller falls back to Topic Brain/static topics.

The trend system never becomes a hard dependency for video generation.

Gemini calls are routed through gemini_client.py.
"""

import os
from datetime import datetime, timedelta, timezone

import requests

try:
    from .gemini_client import generate as gemini_generate
    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from gemini_client import generate as gemini_generate
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False


def _fetch_reddit_trending(
    subreddit: str,
    limit: int = 12,
) -> list:
    """Optional Reddit signal."""

    try:
        resp = requests.get(
            f"https://www.reddit.com/r/{subreddit}/top.json",
            params={
                "t": "week",
                "limit": limit,
            },
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36"
                )
            },
            timeout=10,
        )

        resp.raise_for_status()

        posts = resp.json()["data"]["children"]

        return [
            p["data"]["title"]
            for p in posts
            if not p["data"].get("stickied")
        ]

    except Exception:
        return []


def _fetch_youtube_trending(
    query: str,
    max_results: int = 15,
) -> list:
    """
    Use YouTube Data API search as the primary trend signal.

    Requires:
        YOUTUBE_DATA_API_KEY
    """

    api_key = os.environ.get(
        "YOUTUBE_DATA_API_KEY"
    )

    if not api_key:
        return []

    try:
        published_after = (
            datetime.now(timezone.utc)
            - timedelta(days=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "viewCount",
                "publishedAfter": published_after,
                "maxResults": max_results,
                "key": api_key,
            },
            timeout=15,
        )

        resp.raise_for_status()

        items = resp.json().get("items", [])

        return [
            item["snippet"]["title"]
            for item in items
            if item.get("snippet", {}).get("title")
        ]

    except Exception as e:
        print(
            f"WARNING: YouTube trend fetch failed ({e}); "
            "skipping this signal."
        )

        return []


def _synthesize_topic(
    signal_titles: list,
    config: dict,
) -> str:
    """
    Convert current trend signals into ONE original
    animal/wildlife topic.

    Never copies a source title verbatim.
    """

    signal_text = "\n".join(
        f"- {title}"
        for title in signal_titles[:25]
    )

    prompt = f"""
You are a trend researcher for a high-retention
Animals & Wildlife YouTube channel.

CHANNEL:
{config['display_name']}

NICHE:
Animals & wildlife facts

TONE:
{config['tone']}

These are recent titles getting attention:

{signal_text}

Find the underlying animal or wildlife themes
that are attracting attention.

Then create ONE completely ORIGINAL topic
for this channel.

IMPORTANT:

The topic MUST be about animals or wildlife.

Do NOT create topics about:
- Space
- Planets
- Stars
- Galaxies
- Black holes
- Astronomy
- NASA
- Physics unrelated to animals
- Generic science unrelated to animals

Do NOT copy any title.

Do NOT simply paraphrase a title.

Instead, identify the interesting animal or wildlife
idea behind the trend and create a new,
specific question or fact.

The topic must:

- be factually defensible
- have a surprising answer
- work in a 30-60 second Short
- have strong visual potential
- create a strong curiosity gap
- be understandable to a general audience
- avoid generic school-style facts
- avoid fake mystery
- avoid unsupported speculation
- be interesting enough to tell a friend

DO NOT use phrases such as:

"Scientists are terrified"
"NASA doesn't want you to know"
"This changes everything"
"Scientists can't explain"

unless literally supported by evidence.

Return ONLY the topic itself.

One line.
No quotes.
No explanation.
"""

    topic = (
        gemini_generate(prompt)
        .strip()
        .strip('"')
        .strip("'")
    )

    if not topic or len(topic) > 200:
        raise ValueError(
            f"Unusable synthesized topic: {topic!r}"
        )

    return topic


def get_trending_topic(
    channel_id: str,
    config: dict,
):
    """
    Main entry point.

    Strategy:
        1. YouTube signal
        2. Reddit signal
        3. Gemini synthesis

    Returns None when no reliable signal is available.
    """

    if not config.get("trend_aware"):
        return None

    if not (
        GEMINI_AVAILABLE
        and os.environ.get("GEMINI_API_KEY")
    ):
        return None

    signal = []

    # ---------------------------------------------------------
    # YouTube = PRIMARY SIGNAL
    # ---------------------------------------------------------

    youtube_signal = _fetch_youtube_trending(
        config.get(
            "niche",
            "Animals & wildlife facts",
        )
    )

    if youtube_signal:
        signal.extend(youtube_signal)

        print(
            f"Trend Scout: received "
            f"{len(youtube_signal)} YouTube signals."
        )

    # ---------------------------------------------------------
    # Reddit = OPTIONAL SIGNAL
    # ---------------------------------------------------------

    subreddit = config.get(
        "trend_subreddit"
    )

    if subreddit:
        reddit_signal = _fetch_reddit_trending(
            subreddit
        )

        if reddit_signal:
            signal.extend(reddit_signal)

            print(
                f"Trend Scout: received "
                f"{len(reddit_signal)} Reddit signals."
            )

    # ---------------------------------------------------------
    # No trend signal
    # ---------------------------------------------------------

    if not signal:
        print(
            "Trend Scout: no current signal available; "
            "falling back to Topic Brain/static topics."
        )

        return None

    # Remove duplicates while preserving order.
    signal = list(
        dict.fromkeys(signal)
    )

    # ---------------------------------------------------------
    # Gemini synthesis
    # ---------------------------------------------------------

    try:
        topic = _synthesize_topic(
            signal,
            config,
        )

        print(
            f"Trend Scout: synthesized topic from "
            f"{len(signal)} signal items -> {topic!r}"
        )

        return topic

    except Exception as e:
        print(
            f"WARNING: Trend Scout synthesis failed "
            f"({e}); falling back to Topic Brain/static topics."
        )

        return None