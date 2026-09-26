"""
daily_brain_topics.py

Runs the AI Daily Brain at most once per UTC day per channel and
distributes its ranked winner list across that day's video slots.

Long-form gets winner #1.
Each Short gets the next ranked winner in order.

The Daily Brain is used for animal and wildlife content through the
channel configuration and agent pipeline.

Fails safe at every step. Any failure returns None so generate_script.py
falls back to the existing Trend Scout / static topic selection.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"


def _brain_state_path(channel_id: str) -> Path:
    """Tracks how many of today's ranked winners have been consumed."""
    return STATE_DIR / f"{channel_id}_daily_brain_progress.json"


def _load_todays_report(channel_id: str) -> dict:
    """Load today's saved Daily Brain report, if one exists."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = STATE_DIR / channel_id / "daily_brain" / f"{day}.json"

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _run_daily_brain_once(channel_id: str) -> dict:
    """Run the Daily Brain pipeline when today's report does not exist."""
    from src.agents.agent_manager import run_daily_brain

    return run_daily_brain(channel_id)


def get_daily_brain_topic(
    channel_id: str,
    config: dict,
    is_short: bool,
) -> str:
    """
    Return one topic from today's ranked Daily Brain winners.

    Long-form uses winner #1.
    Shorts use winners #2, #3, #4, etc.

    Returns None on failure or when Daily Brain is disabled.
    """

    if not config.get("use_daily_brain"):
        return None

    report = _load_todays_report(channel_id)

    if not report:
        try:
            print(
                "Daily Brain: no report cached for today yet -- "
                "running now (once per day)."
            )
            report = _run_daily_brain_once(channel_id)

        except Exception as e:
            print(
                f"WARNING: Daily Brain run failed ({e}); "
                "falling back to Trend Scout / static topics."
            )
            return None

    winners = report.get("winners") or []

    if not winners:
        return None

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    progress_path = _brain_state_path(channel_id)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        progress = (
            json.loads(progress_path.read_text())
            if progress_path.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        progress = {}

    if progress.get("date") != day:
        progress = {
            "date": day,
            "long_form_used": False,
            "shorts_used": 0,
        }

    # Long-form always gets the #1 ranked winner.
    if not is_short:
        if progress["long_form_used"]:
            return None

        progress["long_form_used"] = True
        topic = winners[0].get("topic")

    # Shorts use winners #2, #3, #4, etc.
    else:
        index = progress["shorts_used"] + 1

        # If there is only one winner, allow the Short to use it.
        if len(winners) == 1:
            index = 0

        # Do not repeat a winner when the ranked list is exhausted.
        elif index >= len(winners):
            return None

        topic = winners[index].get("topic")
        progress["shorts_used"] += 1

    if not topic:
        return None

    try:
        progress_path.write_text(
            json.dumps(progress, indent=2)
        )
    except OSError:
        return None

    return topic