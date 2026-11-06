"""
Winner Hunter: channel-performance analyst and topic strategist.

Uses real channel analytics to identify repeatable winning patterns,
penalize weak openings/topics, and build a strategy brief for new ideas.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from src.agents.video_registry import recent_videos, record_performance_snapshot, sync_channel_videos
from src.agents.youtube_channel_videos import fetch_channel_videos
from src.analytics_auth import get_analytics_service


MIN_AGE_DAYS = 0.5
LOOKBACK_DAYS = 90
MIN_ELIGIBLE_VIDEOS = 1
MAX_VIDEOS = 5000
ANALYTICS_BATCH_SIZE = 50

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




def _fetch_audience_geography(youtube_analytics) -> dict:
    """Fetch Shorts audience geography and recent country movement."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    base = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "ids": "channel==MINE",
        "metrics": (
            "views,engagedViews,averageViewDuration,"
            "averageViewPercentage,estimatedMinutesWatched"
        ),
        "maxResults": 200,
    }

    try:
        response = youtube_analytics.reports().query(
            **base,
            dimensions="country,creatorContentType",
        ).execute()
    except Exception as exc:
        print(f"WARNING: Winner Hunter country report unavailable ({exc}).")
        return {}

    headers = [h["name"] for h in response.get("columnHeaders", [])]
    countries = {}
    for row in response.get("rows", []):
        item = dict(zip(headers, row))
        # Keep Shorts only. Other formats must not distort the target audience.
        if str(item.get("creatorContentType", "")).upper() != "SHORTS":
            continue
        country = str(item.get("country", "")).upper()
        if not country:
            continue

        def number(key):
            try:
                return float(item.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        countries[country] = {
            "views": number("views"),
            "engaged_views": number("engagedViews"),
            "avg_view_duration": number("averageViewDuration"),
            "retention": number("averageViewPercentage"),
            "watch_minutes": number("estimatedMinutesWatched"),
        }

    total_views = sum(x["views"] for x in countries.values())
    total_engaged = sum(x["engaged_views"] for x in countries.values())
    for item in countries.values():
        item["view_share"] = (
            item["views"] / total_views * 100 if total_views else 0.0
        )
        item["engaged_share"] = (
            item["engaged_views"] / total_engaged * 100
            if total_engaged else 0.0
        )

    # Country is a filter, not a dimension, in the time-based country
    # report. Query the US and UK separately to get valid daily time series.
    trend = {}
    for country in ("US", "GB"):
        try:
            trend_response = youtube_analytics.reports().query(
                ids="channel==MINE",
                startDate=start.strftime("%Y-%m-%d"),
                endDate=end.strftime("%Y-%m-%d"),
                metrics="views",
                dimensions="day,creatorContentType",
                filters=f"country=={country}",
                maxResults=200,
            ).execute()
            trend_headers = [
                h["name"] for h in trend_response.get("columnHeaders", [])
            ]
            country_days = {}
            for row in trend_response.get("rows", []):
                item = dict(zip(trend_headers, row))
                if str(item.get("creatorContentType", "")).upper() != "SHORTS":
                    continue
                day = str(item.get("day", ""))
                try:
                    views = float(item.get("views", 0) or 0)
                except (TypeError, ValueError):
                    views = 0.0
                if day:
                    country_days[day] = views
            trend[country] = country_days
        except Exception as exc:
            print(
                f"WARNING: Winner Hunter {country} daily trend unavailable "
                f"({exc})."
            )

    return {
        "countries": countries,
        "total_views": total_views,
        "trend": trend,
        "period_days": 30,
    }


def _country_weekly_summary(geo: dict, country: str) -> str:
    days = geo.get("trend", {}).get(country, {})
    if not days:
        return "not enough country trend data"
    dates = sorted(days)
    if len(dates) < 8:
        return "not enough daily data for a reliable trend"
    recent = sum(days[d] for d in dates[-7:])
    previous = sum(days[d] for d in dates[-14:-7])
    if previous <= 0:
        return f"last 7 days: {recent:.0f} views; prior week unavailable"
    change = (recent - previous) / previous * 100
    return (
        f"last 7 days {recent:.0f} vs previous 7 days "
        f"{previous:.0f} ({change:+.1f}%)"
    )

def _fetch_traffic_sources(youtube_analytics) -> list[dict]:
    """Summarize where Shorts views came from during the last 30 days."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=30)
    try:
        response = youtube_analytics.reports().query(
            ids="channel==MINE",
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
            metrics="views,engagedViews,estimatedMinutesWatched",
            dimensions="insightTrafficSourceType,creatorContentType",
            maxResults=200,
        ).execute()
    except Exception as exc:
        print(f"WARNING: Winner Hunter traffic-source report unavailable ({exc}).")
        return []

    headers = [h["name"] for h in response.get("columnHeaders", [])]
    sources = {}
    for row in response.get("rows", []):
        item = dict(zip(headers, row))
        if str(item.get("creatorContentType", "")).upper() != "SHORTS":
            continue
        source = str(item.get("insightTrafficSourceType", "UNKNOWN"))
        try:
            views = float(item.get("views", 0) or 0)
            engaged = float(item.get("engagedViews", 0) or 0)
            minutes = float(item.get("estimatedMinutesWatched", 0) or 0)
        except (TypeError, ValueError):
            continue
        current = sources.setdefault(source, {"views": 0.0, "engaged_views": 0.0, "watch_minutes": 0.0})
        current["views"] += views
        current["engaged_views"] += engaged
        current["watch_minutes"] += minutes
    total = sum(item["views"] for item in sources.values())
    result = []
    for source, item in sources.items():
        item["share"] = item["views"] / total * 100 if total else 0.0
        result.append({"source": source, **item})
    return sorted(result, key=lambda item: item["views"], reverse=True)

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


def _fetch_retention_curve(youtube_analytics, video_id: str) -> dict:
    """Fetch the official audience-retention curve for one video."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=LOOKBACK_DAYS)
    try:
        response = youtube_analytics.reports().query(
            ids="channel==MINE",
            startDate=start.strftime("%Y-%m-%d"),
            endDate=end.strftime("%Y-%m-%d"),
            metrics="audienceWatchRatio,relativeRetentionPerformance,startedWatching,stoppedWatching",
            dimensions="elapsedVideoTimeRatio",
            filters=f"video=={video_id}",
            maxResults=200,
        ).execute()
    except Exception as exc:
        print(f"WARNING: Winner Hunter retention report unavailable for {video_id} ({exc}).")
        return {}
    headers = [h["name"] for h in response.get("columnHeaders", [])]
    points = []
    for row in response.get("rows", []):
        item = dict(zip(headers, row))
        try:
            points.append({
                "position": float(item.get("elapsedVideoTimeRatio", 0) or 0),
                "watch_ratio": float(item.get("audienceWatchRatio", 0) or 0),
                "relative": float(item.get("relativeRetentionPerformance", 0) or 0),
                "stopped": float(item.get("stoppedWatching", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    if not points:
        return {}
    peak = max(points, key=lambda point: point["watch_ratio"])
    drop = max(points, key=lambda point: point["stopped"])
    relative_values = [p["relative"] for p in points if p["relative"]]
    return {
        "points": len(points),
        "peak_position": peak["position"] * 100,
        "largest_drop_position": drop["position"] * 100,
        "largest_drop": drop["stopped"],
        "average_relative": sum(relative_values) / len(relative_values) if relative_values else 0.0,
    }

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
        readonly_path = os.environ.get("YOUTUBE_READONLY_TOKEN_PATH")
        if readonly_path:
            channel_videos = fetch_channel_videos(readonly_path)
            sync_channel_videos(channel_id, channel_videos)
            print(f"Winner Hunter: synchronized {len(channel_videos)} channel videos.")
        else:
            print("WARNING: YOUTUBE_READONLY_TOKEN_PATH is not configured; using pipeline registry only.")
        registry = recent_videos(channel_id, limit=MAX_VIDEOS)
    except Exception as exc:
        print(f"WARNING: Winner Hunter could not sync channel videos ({exc}); using registry ({exc}).")
        registry = recent_videos(channel_id, limit=MAX_VIDEOS)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MIN_AGE_DAYS)
    eligible = []

    for entry in registry:
        published = _parse_datetime(entry.get("published_at") or entry.get("recorded_at"))
        if (
            published
            and published <= cutoff
            and entry.get("video_id")
            and (entry.get("topic") or entry.get("title"))
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
    performance = {}
    for offset in range(0, len(video_ids), ANALYTICS_BATCH_SIZE):
        batch = video_ids[offset:offset + ANALYTICS_BATCH_SIZE]
        performance.update(_fetch_performance(analytics, batch))
    if not performance:
        print("Winner Hunter: Analytics returned no per-video rows.")
        return None

    # Keep timestamped snapshots so performance history survives later runs.
    try:
        saved_count = record_performance_snapshot(channel_id, performance)
        print(
            f"Winner Hunter: saved Analytics snapshot for {saved_count} videos."
        )
    except Exception as exc:
        print(f"WARNING: Winner Hunter could not save performance history ({exc}).")

    traffic_sources = _fetch_traffic_sources(analytics)
    geography = _fetch_audience_geography(analytics)

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
            "video_id": video_id,
            "topic": entry.get("topic") or entry.get("title") or "Untitled video",
            **stats,
            "categories": _classify_topic(entry.get("topic") or entry.get("title") or "Untitled video"),
        })

    if not scored:
        return None

    metric_keys = (
        "views",
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

    # Reward both proven reach (views) and current momentum (views/day),
    # while retaining engagement and retention as quality signals.
    weights = {
        "views": 0.25,
        "views_per_day": 0.25,
        "engaged_rate": 0.20,
        "retention": 0.15,
        "subs_per_1000_engaged": 0.05,
        "shares_per_1000_engaged": 0.07,
        "likes_per_1000_engaged": 0.03,
    }

    for item in scored:
        item["performance_score"] = sum(
            _percentile(item[key], metric_values[key]) * weight
            for key, weight in weights.items()
        )

    scored.sort(
        key=lambda item: item["performance_score"], reverse=True
    )
    # Retention reports require one video ID per request. Inspect winners
    # and weak performers, limiting calls to a small representative sample.
    weak = sorted(scored, key=lambda item: item["performance_score"])
    weak = weak[:max(3, min(5, len(weak)))]
    retention_sample = []
    for item in scored[:5] + weak:
        if item not in retention_sample:
            retention_sample.append(item)
    for item in retention_sample:
        item["retention_curve"] = _fetch_retention_curve(
            analytics, item["video_id"]
        )
    patterns = _build_patterns(scored)

    lines = [
        "WINNER HUNTER — CHANNEL GROWTH STRATEGY",
        "",
        "MISSION: Build every Short toward the stretch target of 20,000,000",
        "views within 30 days, regardless of the country where viewers live.",
        "Treat this as an aggressive optimization goal, never a guarantee.",
        "Optimize for global reach, retention, sharing, and repeatable growth.",
        "",
        "HOW TO READ THE DATA:",
        "- Views/day is recent-window view velocity, not lifetime views.",
        "- Engaged-view rate = engagedViews / views. It is an opening-",
        "  engagement proxy, NOT the exact Studio 'viewed vs swiped away' rate.",
        "- Retention is averageViewPercentage.",
        "- Subscriber, share and interaction rates are normalized per 1,000",
        "  engaged views to reduce the advantage of already-large videos.",
        "- Retention curves use official audienceWatchRatio and stoppedWatching",
        "  data. They show where attention drops, not why it drops.",
        "",
        "TOP PERFORMERS — STUDY THEIR TOPIC ANGLE AND VIEWER PROMISE:",
    ]

    # Show the highest raw-view performers first so topic generation
    # can directly favor subjects with demonstrated reach.
    top_by_views = sorted(scored, key=lambda item: item["views"], reverse=True)[:8]
    for item in top_by_views:
        lines.append(
            f'- "{item["topic"]}" | performance score {item["performance_score"]:.0f}/100'
            f" | {int(item['views']):,} views in reporting window"
            f" | {item['views_per_day']:.1f} views/day"
            f" | engaged rate {item['engaged_rate']:.1f}%"
            f" | retention {item['retention']:.1f}%"
            f" | +{item['subs']:.0f} subscribers"
        )

    lines.extend(["", "RETENTION DIAGNOSTICS — TOP AND WEAK SAMPLE:"])
    for item in retention_sample:
        curve = item.get("retention_curve") or {}
        if curve:
            lines.append(
                f'- "{item["topic"]}" | {curve.get("points", 0)} points'
                f" | largest drop near {curve.get('largest_drop_position', 0):.0f}%"
                f" | retention peak near {curve.get('peak_position', 0):.0f}%"
                f" | relative retention avg {curve.get('average_relative', 0):.3f}"
            )

    if traffic_sources:
        lines.extend(["", "SHORTS TRAFFIC SOURCES — LAST 30 DAYS:"])
        for item in traffic_sources[:8]:
            lines.append(
                f"- {item['source']}: {item['views']:,.0f} views "
                f"({item['share']:.1f}%); {item['engaged_views']:,.0f} engaged views"
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

    if geography and geography.get("countries"):
        countries = geography["countries"]
        us = countries.get("US", {})
        gb = countries.get("GB", {})
        total_geo = geography.get("total_views", 0)
        lines.extend([
            "",
            "AUDIENCE GEOGRAPHY — LAST 30 DAYS, SHORTS ONLY:",
            f"- Country-report views total: {total_geo:,.0f}",
            f"- United States: {us.get('view_share', 0):.1f}% of reported views; "
            f"{us.get('views', 0):,.0f} views; "
            f"{us.get('retention', 0):.1f}% average viewed",
            f"- United Kingdom: {gb.get('view_share', 0):.1f}% of reported views; "
            f"{gb.get('views', 0):,.0f} views; "
            f"{gb.get('retention', 0):.1f}% average viewed",
            f"- US trend: {_country_weekly_summary(geography, 'US')}",
            f"- UK trend: {_country_weekly_summary(geography, 'GB')}",
            "- Country reports can omit low-volume data and may not sum to "
            "the channel total.",
            "- Do not overreact to one day's country mix. Use 7-day movement "
            "and the 30-day baseline.",
            "- Treat country data as descriptive analytics only, never as a "
            "targeting restriction or reason to reject a topic.",
            "- Do not favor or exclude any country. Optimize for worldwide "
            "viewer appeal, with clear storytelling that travels across markets.",
        ])

    lines.extend([
        "",
        "STRATEGY RULES FOR THE NEXT TOPIC:",
        "- Choose topics from the full channel evidence; do not impose arbitrary animal-family bans.",
        "- Reuse species, animal families, and subject areas when their videos have high views or strong view velocity.",
        "- Prefer a proven popular subject with a distinct factual angle over an untested subject chosen only for novelty.",
        "- Prefer a clear, instantly understandable visual event.",
        "- Prefer a strong curiosity gap: an unusual action, surprising",
        "  consequence, hidden mechanism, or unexpected survival trick.",
        "- Learn from the best-performing topic patterns, especially those",
        "  that combine strong opening engagement with strong retention.",
        "- Ignore country as a targeting constraint: no country is the goal",
        "  and no country is excluded. Seek the strongest global audience fit.",
        "- Prefer universal visual appeal and clear language that can travel",
        "  across countries; use the configured language naturally.",
        "- A 20M-in-30-days target requires mass appeal, a powerful first-second",
        "  visual, a high-retention story, and strong share potential.",
        "- Avoid topics whose core promise resembles the weak performers.",
        "- Use retention diagnostics to locate likely drop-off moments; improve",
        "  the opening or pacing, but do not assume the graph explains why.",
        "- Use traffic-source mix to separate Shorts Feed discovery from search",
        "  and external traffic when judging each content strategy.",
        "- Avoid generic 'amazing facts' and topics needing long setup.",
        "- Favor subjects that can be explained accurately in 35-55 seconds.",
        "- Never sacrifice factual accuracy for a more clickable claim.",
        "- Do not copy an exact title or repeat the same video premise.",
        "- Reusing a species or proven popular subject is allowed and encouraged when Analytics supports it.",
        "- Popularity and demonstrated view performance outrank novelty.",
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
