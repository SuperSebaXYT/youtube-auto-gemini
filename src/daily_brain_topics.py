
"""
daily_brain_topics.py

Runs the AI Daily Brain at most once per UTC day per channel and
distributes its ranked winner list across that day's video slots.

Long-form gets winner #1.
Each Short gets the next unused ranked winner in order.

Topics are not reused by this module during the same UTC day.
Channel-wide duplicate prevention remains the responsibility of
generate_script.py.

Fails safe at every step. Any failure returns None so generate_script.py
can fall back to Trend Scout / static topic selection.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _brain_state_path(channel_id: str) -> Path:
    """Tracks which Daily Brain winners have been consumed."""
    return STATE_DIR / f"{channel_id}_daily_brain_progress.json"


def _load_todays_report(channel_id: str) -> dict:
    """Load today's saved Daily Brain report, if one exists."""
    path = (
        STATE_DIR
        / channel_id
        / "daily_brain"
        / f"{_today_utc()}.json"
    )

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _run_daily_brain_once(channel_id: str) -> dict:
    """Run the Daily Brain pipeline when today's report does not exist."""
    from src.agents.agent_manager import run_daily_brain

    result = run_daily_brain(channel_id)
    return result if isinstance(result, dict) else {}


def _load_progress(progress_path: Path, day: str) -> dict:
    """Load today's progress, resetting stale or malformed state."""
    try:
        progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if progress_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        progress = {}

    if not isinstance(progress, dict) or progress.get("date") != day:
        return {
            "date": day,
            "long_form_used": False,
            "shorts_used": 0,
            "used_topics": [],
        }

    # Migrate progress files created by older versions.
    progress.setdefault("long_form_used", False)
    progress.setdefault("shorts_used", 0)
    progress.setdefault("used_topics", [])

    if not isinstance(progress["used_topics"], list):
        progress["used_topics"] = []

    try:
        progress["shorts_used"] = max(
            0, int(progress["shorts_used"])
        )
    except (TypeError, ValueError):
        progress["shorts_used"] = 0

    return progress


def _normalize_topic(topic: str) -> str:
    """Normalize a topic for case- and whitespace-insensitive comparison."""
    return " ".join(str(topic).casefold().split())


def get_daily_brain_topic(
    channel_id: str,
    config: dict,
    is_short: bool,
) -> str | None:
    """
    Return one unused topic from today's ranked Daily Brain winners.

    Long-form uses winner #1.
    Shorts use winners #2, #3, #4, etc., skipping already-used winners.
    If only one winner exists, it can be used once, not repeatedly.

    Returns None on failure, exhaustion, or when Daily Brain is disabled.
    """

    if not config.get("use_daily_brain"):
        return None

    report = _load_todays_report(channel_id)

    if not report:
        try:
            print(
                "Daily Brain: no report cached for today yet -- "
                "running now."
            )
            report = _run_daily_brain_once(channel_id)
        except Exception as e:
            print(
                f"WARNING: Daily Brain run failed ({e}); "
                "falling back to Trend Scout / static topics."
            )
            return None

    winners = report.get("winners") or []

    if not isinstance(winners, list) or not winners:
        return None

    # Keep only valid winner records with non-empty topics.
    winners = [
        winner
        for winner in winners
        if isinstance(winner, dict)
        and isinstance(winner.get("topic"), str)
        and winner["topic"].strip()
    ]

    if not winners:
        return None

    day = _today_utc()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = _brain_state_path(channel_id)
    progress = _load_progress(progress_path, day)

    used_topics = {
        _normalize_topic(topic)
        for topic in progress["used_topics"]
        if isinstance(topic, str) and topic.strip()
    }

    if not is_short:
        # Long-form is assigned the top-ranked winner only.
        if progress["long_form_used"]:
            return None

        candidates = winners[:1]
    else:
        # Preserve the ranking policy: Shorts start at winner #2.
        # If there is only one winner, it is eligible once.
        start_index = 1 if len(winners) > 1 else 0
        candidates = winners[start_index:]

    selected_topic = None

    for winner in candidates:
        topic = winner["topic"].strip()
        normalized = _normalize_topic(topic)

        if normalized in used_topics:
            continue

        selected_topic = topic
        break

    if not selected_topic:
        print(
            "Daily Brain: no unused ranked winners remain for this "
            "video slot; falling back to another topic source."
        )
        return None

    # Record the selection so the same Daily Brain topic is not returned
    # again during this UTC day.
    progress["used_topics"] = list(
        dict.fromkeys(
            [
                *progress["used_topics"],
                selected_topic,
            ]
        )
    )

    if is_short:
        progress["shorts_used"] += 1
    else:
        progress["long_form_used"] = True

    try:
        progress_path.write_text(
            json.dumps(progress, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        print(
            f"WARNING: Could not save Daily Brain progress ({e}); "
            "skipping this topic to avoid an untracked repeat."
        )
        return None

    return selected_topic
