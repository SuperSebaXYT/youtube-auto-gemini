"""Animals & Wildlife AI Daily Brain V1.6.

Research layer only. It selects and prepares a winner but does not upload.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.agents.editor_agent import select_winners
from src.agents.memory import load_memory, load_used_topics, remember, save_daily_brain
from src.agents.novelty_agent import check_topics
from src.agents.production_agent import build_brief
from src.agents.science_agent import fact_check_topics
from src.agents.topic_agent import find_topics
from src.trend_scout import _fetch_reddit_trending, _fetch_youtube_trending
from src.channel_analytics import build_performance_summary

ROOT = Path(__file__).resolve().parents[2]


def load_channel_config(channel_id: str) -> dict:
    path = ROOT / "channels" / f"{channel_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Channel config not found: {path}")
    return yaml.safe_load(path.read_text())


def collect_signals(channel_id: str, config: dict) -> list[str]:
    signals: list[str] = []

    yt = _fetch_youtube_trending(
        config.get("niche", "Animals and wildlife facts"),
        max_results=15,
    )

    if yt:
        print(f"Trend Scout: received {len(yt)} YouTube signals.")
        signals.extend(yt)

    subreddit = config.get("trend_subreddit")

    if subreddit:
        reddit = _fetch_reddit_trending(subreddit, limit=12)

        if reddit:
            print(f"Trend Scout: received {len(reddit)} Reddit signals.")
            signals.extend(reddit)

    return list(dict.fromkeys(signals))


def run_daily_brain(channel_id: str) -> dict:
    config = load_channel_config(channel_id)
    memory = load_memory(channel_id)

    previous = list(load_used_topics(channel_id))

    previous.extend(
        str(x.get("topic"))
        for x in memory.get("topics", [])
        if isinstance(x, dict) and x.get("topic")
    )

    previous = list(dict.fromkeys(previous))

    print("=" * 70)
    print(f"{channel_id.upper()} — AI DAILY BRAIN V1.6")
    print("=" * 70)
    print(f"Channel: {config['display_name']}")
    print(f"Previous topics: {len(previous)}")

    signals = collect_signals(channel_id, config)
    print(f"Live signal items: {len(signals)}")

    print("\nSTEP 1 — TOPIC HUNTER")
    candidates = find_topics(
        channel_id,
        config,
        signals,
        previous,
        count=30,
    )

    print(f"Generated candidates: {len(candidates)}")

    if not candidates:
        raise RuntimeError("Topic Hunter returned no usable candidates.")

    candidate_topics = [x["topic"] for x in candidates]

    print("\nSTEP 2 — NOVELTY GUARDIAN")
    novelty_raw = check_topics(
        channel_id,
        candidate_topics,
        previous,
    )

    novelty_by_index = {
        int(x.get("index")) - 1: x
        for x in novelty_raw
        if str(x.get("index", "")).isdigit()
    }

    novelty_pass = []

    for i, item in enumerate(candidates):
        review = novelty_by_index.get(i)

        if review and str(review.get("verdict", "")).upper() == "PASS":
            merged = dict(item)

            merged.update({
                "novelty_score": review.get(
                    "novelty_score",
                    item.get("novelty", 0),
                ),
                "novelty_reason": review.get("reason", ""),
                "similar_previous_topic": review.get(
                    "similar_previous_topic",
                    "",
                ),
            })

            novelty_pass.append(merged)

    print(f"Novelty-approved candidates: {len(novelty_pass)}")

    if not novelty_pass:
        raise RuntimeError("No topic survived novelty screening.")

    science_input = [x["topic"] for x in novelty_pass[:15]]

    print("\nSTEP 3 — SCIENCE GUARDIAN")
    science_raw = fact_check_topics(science_input)

    science_by_index = {
        int(x.get("index")) - 1: x
        for x in science_raw
        if str(x.get("index", "")).isdigit()
    }

    science_pass = []

    for i, item in enumerate(novelty_pass[:15]):
        review = science_by_index.get(i)

        if review and str(review.get("verdict", "")).upper() == "PASS":
            merged = dict(item)

            merged.update({
                "accuracy_score": review.get("accuracy_score", 0),
                "claim_type": review.get("claim_type", ""),
                "safe_framing": review.get("safe_framing", ""),
                "science_risk": review.get("key_risk", ""),
            })

            science_pass.append(merged)

    print(f"Science-approved candidates: {len(science_pass)}")

    if not science_pass:
        raise RuntimeError("No topic survived scientific screening.")

    print("\nSTEP 4 — CHIEF EDITOR")

    token_path = ROOT / "tokens" / f"{channel_id}_analytics_token.pickle"
    client_secret_path = ROOT / "client_secret.json"

    performance_summary = None

    try:
        performance_summary = build_performance_summary(
            channel_id,
            str(token_path),
            str(client_secret_path)
            if client_secret_path.exists()
            else None,
        )
    except Exception as e:
        print(
            f"WARNING: performance summary unavailable ({e}); "
            f"ranking without it."
        )

    editorial = select_winners(
        science_pass,
        config,
        count=9,
        performance_summary=performance_summary,
    )

    winners = (
        editorial.get("winners", [])
        if isinstance(editorial, dict)
        else []
    )

    winner = (
        editorial.get("winner")
        if isinstance(editorial, dict)
        else None
    )

    winner_data = next(
        (x for x in winners if x.get("topic") == winner),
        None,
    )

    if winner_data is None and winners:
        winner_data = winners[0]
        winner = winner_data.get("topic")

    if not winner:
        raise RuntimeError("Chief Editor returned no winner.")

    print("\nSTEP 5 — PRODUCTION DIRECTOR (1 batch call)")

    brief = build_brief(
        winner,
        winner_data or {},
        config,
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    report = {
        "channel_id": channel_id,
        "generated_at": generated_at,
        "signal_count": len(signals),
        "candidate_count": len(candidates),
        "novelty_pass_count": len(novelty_pass),
        "science_pass_count": len(science_pass),
        "winner": winner,
        "winner_data": winner_data,
        "production_brief": brief,
        "strategy_note": (
            editorial.get("strategy_note", "")
            if isinstance(editorial, dict)
            else ""
        ),
        "winners": winners,
        "top_candidates": science_pass,
    }

    report_path = save_daily_brain(
        channel_id,
        report,
    )

    remember(
        channel_id,
        "topics",
        {
            "topic": winner,
            "status": "brain_winner",
            "score": (winner_data or {}).get("score"),
            "date": generated_at,
        },
    )

    remember(
        channel_id,
        "agent_notes",
        {
            "type": "daily_brain",
            "signal_count": len(signals),
            "candidate_count": len(candidates),
            "novelty_pass_count": len(novelty_pass),
            "science_pass_count": len(science_pass),
            "winner": winner,
            "report": str(report_path),
        },
    )

    print("\n" + "=" * 70)
    print("TOP ANIMALS & WILDLIFE OPPORTUNITIES")
    print("=" * 70)

    for w in winners:
        print(
            f"#{w.get('rank', '?')} | "
            f"{w.get('topic', '')} | "
            f"score={w.get('score', '?')} | "
            f"format={w.get('recommended_format', '?')}"
        )

        print(f"    {w.get('why', '')}")

    print("\n" + "=" * 70)
    print(f"WINNER: {winner}")
    print(f"BRAIN REPORT SAVED: {report_path}")
    print("AI DAILY BRAIN COMPLETE — 5 Gemini batch calls maximum")
    print("=" * 70)

    return report


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m "
            "src.agents.agent_manager cosmic_curious"
        )
        raise SystemExit(1)

    run_daily_brain(sys.argv[1])


if __name__ == "__main__":
    main()