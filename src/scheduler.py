"""
scheduler.py

Hourly scheduler for GitHub Actions.

CURRENT MODE:
- Shorts: ENABLED
- Long-form: DISABLED by default

The scheduler checks which Shorts upload slots are due and runs
only ONE job per scheduler run. This prevents multiple Shorts
from being uploaded at the same time if GitHub Actions is delayed.

Channels are controlled by the ENABLED_CHANNELS environment variable.
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


# ============================================================
# MAIN SWITCHES
# ============================================================

ENABLE_SHORTS = True

ENABLE_LONGFORM = False


def _time_to_minutes(time_str: str) -> int:
    """Convert HH:MM into minutes since midnight."""
    h, m = time_str.split(":")
    return int(h) * 60 + int(m)


def _state_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_schedule_state.json"


def _load_schedule_state(channel_id: str, today: str) -> dict:
    """
    Load today's scheduler state.

    A new state is created automatically when the date changes.
    """
    path = _state_path(channel_id)

    if path.exists():
        try:
            data = json.loads(path.read_text())

            if data.get("date") == today:
                return data

        except (json.JSONDecodeError, OSError):
            pass

    return {
        "date": today,
        "done_slots": [],
    }


def _mark_slot_done(
    channel_id: str,
    state: dict,
    slot_id: str,
) -> None:
    """Save a successfully completed slot immediately."""

    if slot_id not in state["done_slots"]:
        state["done_slots"].append(slot_id)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    _state_path(channel_id).write_text(
        json.dumps(state, indent=2)
    )


# ============================================================
# LONG-FORM SUPPORT
# ============================================================
# Kept for future use.
# It is NOT used while ENABLE_LONGFORM = False.


def _longform_last_run_path(channel_id: str) -> Path:
    return STATE_DIR / f"{channel_id}_longform_last_run.json"


def _is_longform_day(
    channel_id: str,
    config: dict,
    today: str,
) -> bool:
    """Check whether Long-form is allowed today."""

    interval_days = config.get("longform_interval_days", 1)

    if interval_days <= 1:
        return True

    path = _longform_last_run_path(channel_id)

    if not path.exists():
        return True

    try:
        last_date_str = json.loads(
            path.read_text()
        ).get("date")

    except (json.JSONDecodeError, OSError):
        return True

    if not last_date_str:
        return True

    last_date = datetime.strptime(
        last_date_str,
        "%Y-%m-%d",
    ).date()

    today_date = datetime.strptime(
        today,
        "%Y-%m-%d",
    ).date()

    days_since = (today_date - last_date).days

    return days_since >= interval_days


def _mark_longform_ran(
    channel_id: str,
    today: str,
) -> None:
    """Save the date of the last successful Long-form upload."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    _longform_last_run_path(channel_id).write_text(
        json.dumps({"date": today})
    )


# ============================================================
# ENABLED CHANNELS
# ============================================================


def _enabled_channel_ids() -> set | None:
    """
    Return enabled channel IDs.

    If ENABLED_CHANNELS is empty, all channels are allowed.
    """

    raw = os.environ.get(
        "ENABLED_CHANNELS",
        "",
    ).strip()

    if not raw:
        return None

    return {
        channel.strip()
        for channel in raw.split(",")
        if channel.strip()
    }


# ============================================================
# SCHEDULER
# ============================================================


def main():

    now = datetime.now(timezone.utc)

    today = now.strftime("%Y-%m-%d")

    now_minutes = (
        now.hour * 60
        + now.minute
    )

    channels_dir = ROOT / "channels"

    enabled = _enabled_channel_ids()

    ran_any = False

    for config_path in sorted(
        channels_dir.glob("*.yaml")
    ):

        try:

            config = yaml.safe_load(
                config_path.read_text()
            )

            if not config:
                print(
                    f"WARNING: Empty config: "
                    f"{config_path.name}"
                )
                continue

            channel_id = config["channel_id"]

            # ------------------------------------------------
            # CHANNEL FILTER
            # ------------------------------------------------

            if (
                enabled is not None
                and channel_id not in enabled
            ):
                continue

            # ------------------------------------------------
            # LOAD STATE
            # ------------------------------------------------

            state = _load_schedule_state(
                channel_id,
                today,
            )

            done_slots = set(
                state["done_slots"]
            )

            jobs_due = []

            # =================================================
            # SHORTS
            # =================================================

            if ENABLE_SHORTS:

                for time_str in config.get(
                    "shorts_upload_times_utc",
                    [],
                ):

                    slot_id = f"short:{time_str}"

                    if (
                        slot_id not in done_slots
                        and _time_to_minutes(
                            time_str
                        ) <= now_minutes
                    ):
                        jobs_due.append(
                            (
                                slot_id,
                                True,
                                time_str,
                            )
                        )

            # =================================================
            # LONG-FORM
            # =================================================

            if ENABLE_LONGFORM:

                upload_time = config.get(
                    "upload_time_utc"
                )

                if upload_time:

                    long_slot_id = (
                        f"long:{upload_time}"
                    )

                    if (
                        long_slot_id not in done_slots
                        and _time_to_minutes(
                            upload_time
                        ) <= now_minutes
                        and _is_longform_day(
                            channel_id,
                            config,
                            today,
                        )
                    ):
                        jobs_due.append(
                            (
                                long_slot_id,
                                False,
                                upload_time,
                            )
                        )

            # =================================================
            # RUN ONLY ONE DUE JOB
            # =================================================

            if jobs_due:

                # Sort by scheduled time and select only
                # the earliest overdue job.
                jobs_due.sort(
                    key=lambda job: _time_to_minutes(
                        job[2]
                    )
                )

                slot_id, is_short, scheduled_time = jobs_due[0]

                ran_any = True

                label = (
                    "SHORT"
                    if is_short
                    else "LONG-FORM"
                )

                print(
                    f"\n### {channel_id} — "
                    f"{label} due "
                    f"(slot {slot_id}, "
                    f"scheduled {scheduled_time} UTC, "
                    f"now {now.strftime('%H:%M')} UTC) "
                    f"###\n"
                )

                cmd = [
                    sys.executable,
                    "-m",
                    "src.main",
                    channel_id,
                ]

                if is_short:
                    cmd.append("--short")

                result = subprocess.run(
                    cmd
                )

                # ------------------------------------------------
                # FAILED JOB
                # ------------------------------------------------

                if result.returncode != 0:

                    print(
                        f"WARNING: {channel_id} "
                        f"{label} failed "
                        f"(exit {result.returncode}). "
                        f"Slot will be retried "
                        f"on the next scheduler run.",
                        file=sys.stderr,
                    )

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                else:

                    _mark_slot_done(
                        channel_id,
                        state,
                        slot_id,
                    )

                    if not is_short:
                        _mark_longform_ran(
                            channel_id,
                            today,
                        )

        except Exception as exc:

            print(
                f"ERROR processing "
                f"{config_path.name}: {exc}",
                file=sys.stderr,
            )

            continue

    # ============================================================
    # NOTHING TO DO
    # ============================================================

    if not ran_any:

        print(
            f"Nothing due at "
            f"{now.strftime('%H:%M')} UTC. "
            f"Nothing to do."
        )


if __name__ == "__main__":
    main()