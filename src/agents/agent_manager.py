"""Animals & Wildlife AI Daily Brain V1.6.

Research layer only. It selects and prepares winners but does not upload.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.agents.editor_agent import select_winners
from src.agents.memory import (
    load_memory,
    load_used_topics,
    remember,
    save_daily_brain,
)
from src.agents.novelty_agent import check_topics
from src.agents.production_agent import build_brief
from src.agents.science_agent import fact_check_topics
from src.agents.topic_agent import find_topics
from src.trend_scout import (
    _fetch_reddit_trending,
    _fetch_youtube_trending,
)
from src.channel_analytics import build_performance_summary


ROOT = Path(__file__).resolve().parents[2]


def load_channel_config(channel_id: str) -> dict:
    path = ROOT / "channels" / f"{channel_id}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Channel config not found: {path}")

    return yaml.safe_load(path.read_text())


def collect_signals(channel_id: str, config: dict) -> list[str]:
    signals: list[str] = []

    niche = config.get(
        "niche",
        "Animals and wildlife facts",
    )

    yt = _fetch_youtube_trending(
        niche,
        max_results=15,
    )

    if yt:
        print(
            f"Trend Scout: received {len(yt)} "
            "YouTube signals."
        )
        signals.extend(yt)

    subreddit = config.get("trend_subreddit")

    if subreddit:
        reddit = _fetch_reddit_trending(
            subreddit,
            limit=12,
        )

        if reddit:
            print(
                f"Trend Scout: received {len(reddit)} "
                "Reddit signals."
            )
            signals.extend(reddit)

    return list(dict.fromkeys(signals))


def run_daily_brain(channel_id: str) -> dict:
    config = load_channel_config(channel_id)

    memory = load_memory(channel_id)

    previous = list(load