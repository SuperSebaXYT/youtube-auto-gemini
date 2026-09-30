"""
winner_hunter.py

Analyzes the channel's real historical performance and extracts
winning content patterns for the next topic-generation cycle.

It does NOT copy previous videos.
It identifies patterns such as:
- animal types
- abilities
- behaviors
- survival mechanisms
- dangerous/cute/unusual angles
- hook styles
- recurring subject patterns

Fails safely: if Analytics is unavailable, it returns an empty
performance context and never blocks video production.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from src.agents.video_registry import recent_videos
from src.analytics_auth import get_analytics_service


MIN_AGE_DAYS = 3
LOOKBACK_DAYS = 90
MIN_ELIGIBLE_VIDEOS = 5
MAX_VIDEOS = 200


def _get_token_path(channel_id: str) -> str | None:
    """
    Finds the YouTube Analytics OAuth token path.

    Preferred:
        YOUTUBE_ANALYTICS_TOKEN_PATH

    Optional channel-specific:
        YOUTUBE_ANALYTICS_TOKEN_PATH_<CHANNEL_ID>
    """
    channel_key = channel_id.upper().replace("-", "_")

    channel_specific = os.environ.get(
        f"YOUTUBE_ANALYTICS_TOKEN_PATH_{channel_key}"
    )
    if channel_specific:
        return channel_specific

    return os.environ.get("YOUTUBE_ANALYTICS_TOKEN_PATH")


def _fetch_performance(youtube_analytics, video_ids: list[str]) -> dict:
    if not video_ids:
        return {}

    try:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=LOOKBACK_DAYS)

        response = youtube_analytics.reports().query(
            ids="channel==MINE",
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
            metrics="views,averageViewPercentage,subscribersGained",
            dimensions="video",
            filters=f"video=={','.join(video_ids)}",
            maxResults=MAX_VIDEOS,
        ).execute()

        rows = response.get("rows", [])
        headers = [
            header["name"]
            for header in response.get("columnHeaders", [])
        ]

        result = {}

        for row in rows:
            item = dict(zip(headers, row))
            video_id = item.get("video")

            if not video_id:
                continue

            result[video_id] = {
                "views": float(item.get("views", 0) or 0),
                "retention": float(
                    item.get("averageViewPercentage", 0) or 0
                ),
                "subs": float(
                    item.get("subscribersGained", 0) or 0
                ),
            }

        return result

    except Exception as exc:
        print(
            f"WARNING: Winner Hunter Analytics query failed "
            f"({exc}); skipping performance learning this run."
        )
        return {}


def _percentile(value: float, values: list[float]) -> float:
    if not values:
        return 0.0

    if len(values) == 1:
        return 100.0

    below_or_equal = sum(1 for x in values if x <= value)

    return (below_or_equal / len(values)) * 100.0


def _classify_topic(topic: str) -> list[str]:
    """
    Lightweight semantic classification.

    This is intentionally rule-based so the Winner Hunter does not
    waste an AI request just to classify every old video.
    """
    text = topic.lower()

    patterns = {
        "extreme abilities": [
            "can ",
            "ability",
            "strong",
            "fast",
            "power",
            "survive",
            "regrow",
            "regenerate",
        ],
        "strange behavior": [
            "weird",
            "strange",
            "weirdest",
            "behavior",
            "behaviour",
            "why do",
            "why does",
        ],
        "dangerous animals": [
            "dangerous",
            "deadly",
            "venom",
            "poison",
            "bite",
            "attack",
            "killer",
        ],
        "animal intelligence": [
            "smart",
            "intelligent",
            "intelligence",
            "memory",
            "learn",
            "problem solving",
        ],
        "extreme survival": [
            "survive",
            "survival",
            "extreme",
            "cold",
            "heat",
            "desert",
            "deep sea",
            "ocean",
        ],
        "cute but surprising": [
            "cute",
            "adorable",
            "baby",
            "tiny",
            "small",
        ],
        "animal senses": [
            "see",
            "vision",
            "hear",
            "hearing",
            "smell",
            "sense",
            "taste",
        ],
        "predators and prey": [
            "predator",
            "prey",
            "hunt",
            "hunting",
        ],
        "rare animals": [
            "rare",
            "unknown",
            "little known",
            "rarely",
        ],
        "animal anatomy": [
            "body",
            "heart",
            "brain",
            "teeth",
            "tongue",
            "eyes",
            "skin",
            "bones",
        ],
    }

    categories = []

    for category, keywords in patterns.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)

    return categories or ["general animal facts"]


def _build_patterns(scored: list[dict]) -> list[tuple[str, float, int]]:
    """
    Aggregates performance by semantic topic pattern.
    """
    groups: dict[str, list[dict]] = {}

    for item in scored:
        for category in item["categories"]:
            groups.setdefault(category, []).append(item)

    patterns = []

    for category, items in groups.items():
        if not items:
            continue

        avg_score = sum(
            item["performance_score"]
            for item in items
        ) / len(items)

        patterns.append(
            (
                category,
                avg_score,
                len(items),
            )
        )

    patterns.sort(key=lambda x: x[1], reverse=True)

    return patterns


def build_winner_context(channel_id: str) -> str | None:
    """
    Main entry point.

    Returns a compact prompt-ready block describing which kinds of
    animal content have historically performed best.

    Returns None when there is not enough usable Analytics data.
    """

    try:
        registry = recent_videos(
            channel_id,
            limit=MAX_VIDEOS,
        )
    except Exception as exc:
        print(
            f"WARNING: Winner Hunter could not read registry "
            f"({exc}); skipping."
        )
        return None

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=MIN_AGE_DAYS)
    )

    eligible = []

    for entry in registry:
        recorded_at = entry.get("recorded_at")
        video_id = entry.get("video_id")
        topic = entry.get("topic")

        if not (recorded_at and video_id and topic):
            continue

        try:
            published = datetime.fromisoformat(recorded_at)
        except (ValueError, TypeError):
            continue

        if published <= cutoff:
            eligible.append(entry)

    if len(eligible) < MIN_ELIGIBLE_VIDEOS:
        print(
            f"Winner Hunter: only {len(eligible)} eligible videos; "
            f"need at least {MIN_ELIGIBLE_VIDEOS}."
        )
        return None

    token_path = _get_token_path(channel_id)

    if not token_path:
        print(
            "Winner Hunter: YOUTUBE_ANALYTICS_TOKEN_PATH is not "
            "configured; skipping performance learning."
        )
        return None

    try:
        youtube_analytics = get_analytics_service(
            token_path,
            os.environ.get("YOUTUBE_CLIENT_SECRET_PATH"),
        )
    except Exception as exc:
        print(
            f"WARNING: Winner Hunter Analytics authentication failed "
            f"({exc}); skipping."
        )
        return None

    video_ids = [
        entry["video_id"]
        for entry in eligible
    ]

    performance = _fetch_performance(
        youtube_analytics,
        video_ids,
    )

    if not performance:
        return None

    views = [
        stats["views"]
        for stats in performance.values()
    ]

    retentions = [
        stats["retention"]
        for stats in performance.values()
    ]

    subs = [
        stats["subs"]
        for stats in performance.values()
    ]

    scored = []

    for entry in eligible:
        video_id = entry["video_id"]

        if video_id not in performance:
            continue

        stats = performance[video_id]

        view_percentile = _percentile(
            stats["views"],
            views,
        )

        retention_percentile = _percentile(
            stats["retention"],
            retentions,
        )

        sub_percentile = _percentile(
            stats["subs"],
            subs,
        )

        # Balanced score:
        # views = reach
        # retention = viewer satisfaction
        # subscribers = conversion
        performance_score = (
            view_percentile * 0.45
            + retention_percentile * 0.35
            + sub_percentile * 0.20
        )

        categories = _classify_topic(
            entry["topic"]
        )

        scored.append(
            {
                "topic": entry["topic"],
                "views": stats["views"],
                "retention": stats["retention"],
                "subs": stats["subs"],
                "performance_score": performance_score,
                "categories": categories,
            }
        )

    if not scored:
        return None

    scored.sort(
        key=lambda item: item["performance_score"],
        reverse=True,
    )

    patterns = _build_patterns(scored)

    top_videos = scored[:8]

    lines = [
        "WINNER HUNTER — REAL CHANNEL PERFORMANCE",
        "",
        "Use these results to guide NEW topics.",
        "Do NOT copy existing titles or topics.",
        "",
        "TOP PERFORMING VIDEOS:",
    ]

    for item in top_videos:
        lines.append(
            f'- "{item["topic"]}" — '
            f'{int(item["views"]):,} views, '
            f'{item["retention"]:.0f}% retention, '
            f'{int(item["subs"]):,} subscribers gained'
        )

    if patterns:
        lines.extend(
            [
                "",
                "WINNING CONTENT PATTERNS:",
            ]
        )

        for category, score, sample_count in patterns[:8]:
            lines.append(
                f"- {category}: "
                f"performance index {score:.0f}/100 "
                f"across {sample_count} video(s)"
            )

    weak = sorted(
        scored,
        key=lambda item: item["performance_score"],
    )[:5]

    if weak:
        lines.extend(
            [
                "",
                "WEAKER PATTERNS TO AVOID REPEATING:",
            ]
        )

        for item in weak:
            lines.append(
                f'- "{item["topic"]}" — '
                f'{int(item["views"]):,} views, '
                f'{item["retention"]:.0f}% retention'
            )

    lines.extend(
        [
            "",
            "IMPORTANT:",
            "- Generate NEW subjects inspired by successful patterns.",
            "- Never copy a previous title.",
            "- Never make a near-duplicate of a previous topic.",
            "- Keep every topic about real animals or wildlife.",
            "- A winning pattern is a signal, not a guarantee.",
        ]
    )

    result = "\n".join(lines)

    print(
        f"Winner Hunter: analyzed {len(scored)} videos "
        f"and found {len(patterns)} content patterns."
    )

    return result