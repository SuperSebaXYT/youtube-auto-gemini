"""
Winner Hunter: channel-performance analyst and topic strategist.

Uses real channel analytics to identify repeatable winning patterns,
penalize weak openings/topics, and build a strategy brief for new ideas.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from src.agents.video_registry import recent_videos
from src.analytics_auth import get_analytics_service


MIN_AGE_DAYS = 0.5
LOOKBACK_DAYS = 90
MIN_ELIGIBLE_VIDEOS = 5
MAX_VIDEOS = 200

FELID_TERMS = (
    "cat", "cats", "kitten", "kittens", "feline", "bobcat",
    "lynx", "lion", "tiger", "leopard", "jaguar", "cheetah",
    "cougar", "puma", "ocelot", "caracal", "serval",
)


def _get_token_path(channel_id: str) -> str | None:
    channel_key = channel_id.upper().replace("-", "_")
    channel_specific = os.environ.get(
        f"YOUTUBE_ANALYTICS_TOKEN_PATH_{channel_key}"
    )
    return channel_specific or os.environ.get("YOUTUBE_ANALYTICS_TOKEN_PATH")


def _parse_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _query_report(youtube_analytics, video_ids: list[str], metrics: str):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    return youtube_analytics.reports().query(
        ids="channel==MINE",
        startDate=start.strftime("%Y-%m-%d"),
        endDate=end.strftime("%Y-%m-%d"),
        metrics=metrics,
        dimensions="video",
        filters=f"video=={','.join(video_ids)}",
        maxResults=MAX_VIDEOS,
    ).execute()


def _fetch_performance(youtube_analytics, video_ids: list[str]) -> dict:
    if not video_ids:
        return {}

    # engagedViews is an API-supported signal for views that passed the
    # first frame / initial seconds or were deliberately opened. It is
    # useful as an opening-engagement proxy, not an exact Studio swipe rate.
    metric_sets = [
        "views,engagedViews,averageViewPercentage,averageViewDuration,"
        "subscribersGained,likes,shares,comments",
        "views,averageViewPercentage,subscribersGained",
    ]

    for metric_set in metric_sets:
        try:
            response = _query_report(
                youtube_analytics, video_ids, metric_set
            )
            headers = [
                header["name"]
                for header in response.get("columnHeaders", [])
            ]
            result = {}

            for row in response.get("rows", []):
                item = dict(zip(headers, row))
                video_id = item.get("video")
                if not video_id:
                    continue

                def number(key):
                    try:
                        return float(item.get(key, 0) or 0)
                    except (ValueError, TypeError):
                        return 0.0

                views = number("views")
                engaged = number("engagedViews")
                result[video_id] = {
                    "views": views,
                    "engaged_views": engaged,
                    "engaged_rate": min(100.0, engaged / views * 100.0)
                    if views > 0 and engaged > 0 else 0.0,
                    "retention": number("averageViewPercentage"),
                    "avg_duration": number("averageViewDuration"),
                    "subs": number("subscribersGained"),
                    "likes": number("likes"),
                    "shares": number("shares"),
                    "comments": number("comments"),
                }

            if result:
                return result

        except Exception as exc:
            print(
                f"WARNING: Winner Hunter Analytics query failed for "
                f"'{metric_set}' ({exc})."
            )

    print("WARNING: Winner Hunter could not retrieve usable Analytics rows.")
    return {}


def _percentile(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item <= value) / len(values) * 100.0


def _classify_topic(topic: str) -> list[str]:
    text = topic.lower()
    patterns = {
        "extreme abilities": [
            "can ", "ability", "strong", "fast", "power", "regrow",
            "regenerate", "shoot", "survive",
        ],
        "strange behavior": [
            "weird", "strange", "behavior", "behaviour", "why do",
            "why does", "learned", "use ",
        ],
        "dangerous animals": [
            "dangerous", "deadly", "venom", "poison", "bite", "attack",
            "killer",
        ],
        "animal intelligence": [
            "smart", "intelligent", "intelligence", "memory", "learn",
            "problem solving",
        ],
        "extreme survival": [
            "survive", "survival", "extreme", "cold", "heat", "desert",
            "deep sea", "ocean", "snow",
        ],
        "animal senses": [
            "see", "vision", "blind", "hear", "hearing", "smell",
            "sense", "navigate",
        ],
        "predators and prey": ["predator", "prey", "hunt", "hunting"],
        "rare animals": ["rare", "unknown", "little known", "rarely"],
        "animal anatomy": [
            "body", "heart", "brain", "teeth", "tongue", "eyes", "skin",
            "bones", "feet", "paws",
        ],
        "animal engineering": [
            "build", "built", "structure", "nest", "dam", "web", "colony",
            "architecture",
        ],
    }
    categories = [
        category for category, keywords in patterns.items()
        if any(keyword in text for keyword in keywords)
    ]
    return categories or ["general animal facts"]


def _build_patterns(scored: list[dict]) -> list[tuple[str, float, int]]:
    groups: dict[str, list[dict]] = {}
    for item in scored:
        for category in item["categories"]:
            groups.setdefault(category, []).append(item)

    patterns = []
    for category, items in groups.items():
        average = sum(
            item["performance_score"] for item in items
        ) / len(items)
        patterns.append((category, average, len(items)))

    return sorted(patterns, key=lambda item: item[1], reverse=True)


def _has_felid_topic(topic_text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:" + "|".join(map(re.escape, FELID_TERMS)) + r")\b",
            topic_text.lower(),
        )
    )


def build_winner_context(channel_id: str) -> str | None:
    """Return a data-backed strategy brief for the topic generator."""
    try:
        registry = recent_videos(channel_id, limit=MAX_VIDEOS)
    except Exception as exc:
        print(f"WARNING: Winner Hunter could not read registry ({exc}).")
        return None

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MIN_AGE_DAYS)
    eligible = []

    for entry in registry:
        published = _parse_datetime(entry.get("recorded_at"))
        if (
            published
            and published <= cutoff
            and entry.get("video_id")
            and entry.get("topic")
        ):
            entry = dict(entry)
            entry["_published_at"] = published
            eligible.append(entry)

    if len(eligible) < MIN_ELIGIBLE_VIDEOS:
        print(
            f"Winner Hunter: only {len(eligible)} eligible videos; "
            f"need at least {MIN_ELIGIBLE_VIDEOS}."
        )
        return None

    token_path = _get_token_path(channel_id)
    if not token_path:
        print("Winner Hunter: Analytics token path is not configured.")
        return None

    try:
        analytics = get_analytics_service(
            token_path,
            os.environ.get("YOUTUBE_CLIENT_SECRET_PATH"),
        )
    except Exception as exc:
        print(f"WARNING: Winner Hunter Analytics authentication failed ({exc}).")
        return None

    video_ids = list(dict.fromkeys(
        entry["video_id"] for entry in eligible
    ))
    performance = _fetch_performance(analytics, video_ids)
    if not performance:
        return None

    scored = []
    for entry in eligible:
        video_id = entry["video_id"]
        stats = performance.get(video_id)
        if not stats:
            continue

        age_days = max(
            MIN_AGE_DAYS,
            (now - entry["_published_at"]).total_seconds() / 86400.0,
        )
        # Analytics is queried over the last 90 days. Cap the denominator
        # so older uploads are compared on the same reporting window.
        effective_age = min(age_days, float(LOOKBACK_DAYS))
        stats = dict(stats)
        stats["views_per_day"] = stats["views"] / effective_age
        engaged = max(stats["engaged_views"], 1.0)
        stats["subs_per_1000_engaged"] = stats["subs"] / engaged * 1000
        stats["shares_per_1000_engaged"] = stats["shares"] / engaged * 1000
        stats["likes_per_1000_engaged"] = (
            stats["likes"] + stats["comments"]
        ) / engaged * 1000

        scored.append({
            "topic": entry["topic"],
            **stats,
            "categories": _classify_topic(entry["topic"]),
        })

    if not scored:
        return None

    metric_keys = (
        "views_per_day",
        "engaged_rate",
        "retention",
        "subs_per_1000_engaged",
        "shares_per_1000_engaged",
        "likes_per_1000_engaged",
    )
    metric_values = {
        key: [item[key] for item in scored]
        for key in metric_keys
    }

    # Opening engagement and retention matter more than raw views alone.
    weights = {
        "views_per_day": 0.25,
        "engaged_rate": 0.25,
        "retention": 0.25,
        "subs_per_1000_engaged": 0.10,
        "shares_per_1000_engaged": 0.10,
        "likes_per_1000_engaged": 0.05,
    }

    for item in scored:
        item["performance_score"] = sum(
            _percentile(item[key], metric_values[key]) * weight
            for key, weight in weights.items()
        )

    scored.sort(
        key=lambda item: item["performance_score"], reverse=True
    )
    patterns = _build_patterns(scored)
    weak = sorted(
        scored, key=lambda item: item["performance_score"]
    )[:max(3, min(5, len(scored)))]

    lines = [
        "WINNER HUNTER — CHANNEL GROWTH STRATEGY",
        "",
        "MISSION: Think like an experienced YouTube creator whose job is to",
        "find repeatable topics with breakout potential, not merely generate",
        "interesting animal facts. No topic can guarantee viral results.",
        "",
        "HOW TO READ THE DATA:",
        "- Views/day is recent-window view velocity, not lifetime views.",
        "- Engaged-view rate = engagedViews / views. It is an opening-",
        "  engagement proxy, NOT the exact Studio 'viewed vs swiped away' rate.",
        "- Retention is averageViewPercentage.",
        "- Subscriber, share and interaction rates are normalized per 1,000",
        "  engaged views to reduce the advantage of already-large videos.",
        "",
        "TOP PERFORMERS — STUDY THEIR TOPIC ANGLE AND VIEWER PROMISE:",
    ]

    for item in scored[:8]:
        lines.append(
            f'- "{item["topic"]}" | score {item["performance_score"]:.0f}/100'
            f" | {int(item['views']):,} views in reporting window"
            f" | {item['views_per_day']:.1f} views/day"
            f" | engaged rate {item['engaged_rate']:.1f}%"
            f" | retention {item['retention']:.1f}%"
            f" | +{item['subs']:.0f} subscribers"
        )

    if patterns:
        lines.extend(["", "WINNING CONTENT PATTERNS:"])
        for category, score, count in patterns[:10]:
            lines.append(
                f"- {category}: index {score:.0f}/100 "
                f"from {count} video(s)"
            )

    lines.extend(["", "WEAK PERFORMERS — DO NOT REPEAT THEIR CORE PROMISE:"])
    for item in weak:
        lines.append(
            f'- "{item["topic"]}" | score {item["performance_score"]:.0f}/100'
            f" | engaged rate {item['engaged_rate']:.1f}%"
            f" | retention {item['retention']:.1f}%"
        )

    weak_text = " ".join(item["topic"] for item in weak)
    if _has_felid_topic(weak_text):
        lines.extend([
            "",
            "AVOID UNDERPERFORMING SUBJECT FAMILY:",
            "- FELIDS: domestic cats, kittens, bobcats, lynx, lions, tigers,",
            "  leopards, jaguars, cheetahs, cougars and other cats.",
            "- Do not suggest another felid under a different name or angle.",
        ])

    lines.extend([
        "",
        "STRATEGY RULES FOR THE NEXT TOPIC:",
        "- Find a NEW animal and a NEW specific behavior or ability.",
        "- Prefer a clear, instantly understandable visual event.",
        "- Prefer a strong curiosity gap: an unusual action, surprising",
        "  consequence, hidden mechanism, or unexpected survival trick.",
        "- Learn from the best-performing topic patterns, especially those",
        "  that combine strong opening engagement with strong retention.",
        "- Avoid topics whose core promise resembles the weak performers.",
        "- Avoid generic 'amazing facts' and topics needing long setup.",
        "- Favor subjects that can be explained accurately in 35-55 seconds.",
        "- Never sacrifice factual accuracy for a more clickable claim.",
        "- Never copy an existing title or near-duplicate a used topic.",
        "",
        "A topic is promising only when it has: a 1-second visual hook,",
        "a specific factual payoff, visual footage potential, broad curiosity,",
        "and enough verified material for a concise story.",
    ])

    print(
        f"Winner Hunter: scored {len(scored)} videos, "
        f"identified {len(patterns)} patterns; "
        f"best score {scored[0]['performance_score']:.1f}/100."
    )
    return "\n".join(lines)
