"""
scheduler.py
Runs hourly (triggered by GitHub Actions cron). For each ENABLED channel,
checks whether each configured upload slot is due *and hasn't already run
today*, then runs main.py for exactly what's due — 1 long-form video at
its scheduled hour (optionally gated to every N days via
config["longform_interval_days"]), and up to 1 short at each of its
scheduled short hours.

Why "due" isn't just "current hour == slot hour": GitHub's cron can be
delayed or occasionally drop a tick entirely during high load — an exact
hour-match would silently and permanently lose that day's upload if the
tick for that specific hour never landed. Instead, a slot is due once the
current time has reached it AND it hasn't already run today (tracked in
state/{channel_id}_schedule_state.json, which persists across runs via the
same GitHub Actions cache already used for topic-dedup state). This means
a late or dropped tick just catches up on the next tick instead of losing
the slot for the day, while a per-day state reset means slots still only
ever fire once per day, not once per tick.

Which channels actually run is controlled by the ENABLED_CHANNELS env var
(comma-separated channel_ids, e.g. "mind_bites"). This lets you bring
channels online one at a time as you finish setting up their YouTube
tokens, instead of the scheduler trying (and failing) to upload to
channels that aren't authorized yet. If ENABLED_CHANNELS is unset, ALL
channels in channels/*.yaml run — only unset it once every channel has
a real token.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"


def _time_to_minutes(time_str: str) -> int:
    """time_str is HH:MM -> minutes since midnight, for due/not-due comparison."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _state_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_schedule_state.json"


def _load_schedule_state(channel_id: str, today: str) -> dict:
    """Loads today's set of already-run slot ids for this channel. If the
    stored state is from a previous day (or doesn't exist yet), starts
    fresh — this is what makes slots eligible to run again each new day."""
    path = _state_path(channel_id)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/missing state — treat as a fresh day
    return {"date": today, "done_slots": []}


def _mark_slot_done(channel_id: str, state: dict, slot_id: str) -> None:
    """Persists immediately after each job (not batched at the end) so
    that if a later job in the same run crashes, earlier ones in this run
    are still correctly recorded as done and won't re-fire next tick."""
    state["done_slots"].append(slot_id)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(channel_id).write_text(json.dumps(state))


def _longform_last_run_path(channel_id: str) -> Path:
    """Separate from the daily schedule_state (which intentionally resets
    every day) -- this tracks the last date long-form actually succeeded,
    persisting ACROSS day boundaries, since that's what an every-N-days
    gate needs to check."""
    return STATE_DIR / f"{channel_id}_longform_last_run.json"


def _is_longform_day(channel_id: str, config: dict, today: str) -> bool:
    """True if long-form is allowed to run today. Daily (interval=1) by
    default for backward compatibility -- channels that don't set
    longform_interval_days behave exactly as before. With interval=2
    ("every other day"), skips today if it last succeeded yesterday (or
    today already, though the daily done_slots check handles that case
    too), and runs if 2+ days have passed or it's never run before."""
    interval_days = config.get("longform_interval_days", 1)
    if interval_days <= 1:
        return True

    path = _longform_last_run_path(channel_id)
    if not path.exists():
        return True  # never run before -- always eligible
    try:
        last_date_str = json.loads(path.read_text()).get("date")
    except (json.JSONDecodeError, OSError):
        return True  # corrupt state -- fail open rather than silently skip forever

    if not last_date_str:
        return True

    last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
    today_date = datetime.strptime(today, "%Y-%m-%d").date()
    days_since = (today_date - last_date).days
    return days_since >= interval_days


def _mark_longform_ran(channel_id: str, today: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _longform_last_run_path(channel_id).write_text(json.dumps({"date": today}))


def _enabled_channel_ids() -> set | None:
    """Returns None if all channels are enabled, else a set of allowed ids."""
    raw = os.environ.get("ENABLED_CHANNELS", "").strip()
    if not raw:
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


def main():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    now_minutes = now.hour * 60 + now.minute
    channels_dir = ROOT / "channels"
    enabled = _enabled_channel_ids()
    ran_any = False

    for config_path in sorted(channels_dir.glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        channel_id = config["channel_id"]

        if enabled is not None and channel_id not in enabled:
            continue  # not set up yet — skip silently

        state = _load_schedule_state(channel_id, today)
        done_slots = set(state["done_slots"])

        jobs_due = []  # list of (slot_id, is_short)
        
        for t in config["shorts_upload_times_utc"]:
            short_slot_id = f"short:{t}"
            if short_slot_id not in done_slots and _time_to_minutes(t) <= now_minutes:
                jobs_due.append((short_slot_id, True))

        for slot_id, is_short in jobs_due:
            ran_any = True
            label = "SHORT" if is_short else "LONG-FORM"
            print(f"\n### {channel_id} — {label} due (slot {slot_id}, now {now.strftime('%H:%M')} UTC) ###\n")
            cmd = [sys.executable, "-m", "src.main", channel_id]
            if is_short:
                cmd.append("--short")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"WARNING: {channel_id} {label} run failed (exit {result.returncode}); "
                      f"NOT marking slot done, so it will retry next tick instead of being lost.",
                      file=sys.stderr)
                # Deliberately don't sys.exit here — one channel failing
                # shouldn't block the others in the same hourly tick.
                continue

            _mark_slot_done(channel_id, state, slot_id)
            if not is_short:
                _mark_longform_ran(channel_id, today)

    if not ran_any:
        print(f"Nothing due at {now.strftime('%H:%M')} UTC. Nothing to do.")


if __name__ == "__main__":
    main()
