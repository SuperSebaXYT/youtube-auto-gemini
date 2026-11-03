"""Persistent registry and analytics history for Cosmic Curious."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"


def _channel_dir(channel_id: str) -> Path:
    return STATE_DIR / channel_id


def _path(channel_id: str) -> Path:
    return _channel_dir(channel_id) / "video_registry.json"


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def load_registry(channel_id: str) -> list[dict]:
    return _read_json_list(_path(channel_id))


def record_video(channel_id: str, record: dict) -> dict:
    folder = _channel_dir(channel_id)
    folder.mkdir(parents=True, exist_ok=True)
    rows = load_registry(channel_id)
    entry = dict(record)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry.setdefault("recorded_at", timestamp)
    # This is the time the successful upload was recorded by our pipeline,
    # not a claim that YouTube supplied an exact publish timestamp.
    entry.setdefault("uploaded_at", entry["recorded_at"])
    rows.append(entry)
    rows = rows[-1000:]
    _path(channel_id).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return entry


def record_performance_snapshot(
    channel_id: str,
    performance: dict[str, dict],
) -> int:
    """Append one timestamped Analytics snapshot per video."""
    if not performance:
        return 0
    folder = _channel_dir(channel_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "performance_history.json"
    rows = _read_json_list(path)
    captured_at = datetime.now(timezone.utc).isoformat()
    for video_id, metrics in performance.items():
        rows.append({
            "video_id": video_id,
            "captured_at": captured_at,
            "metrics": metrics,
        })
    rows = rows[-10000:]
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(performance)


def recent_videos(channel_id: str, limit: int = 50) -> list[dict]:
    return load_registry(channel_id)[-limit:]


def recent_performance_snapshots(
    channel_id: str,
    limit: int = 500,
) -> list[dict]:
    return _read_json_list(
        _channel_dir(channel_id) / "performance_history.json"
    )[-limit:]



def sync_channel_videos(channel_id: str, videos: list[dict]) -> int:
    """Merge Data API channel metadata without overwriting pipeline-specific fields."""
    if not videos:
        return 0
    folder = _channel_dir(channel_id)
    folder.mkdir(parents=True, exist_ok=True)
    existing = load_registry(channel_id)
    by_id = {
        str(item.get("video_id")): dict(item)
        for item in existing
        if item.get("video_id")
    }
    for video in videos:
        video_id = str(video.get("video_id") or "")
        if not video_id:
            continue
        merged = by_id.get(video_id, {})
        # Preserve topic/script/source and pipeline timestamps; refresh YouTube metadata.
        for key, value in video.items():
            if value is not None:
                merged[key] = value
        by_id[video_id] = merged
    rows = list(by_id.values())
    rows.sort(key=lambda item: str(item.get("published_at") or item.get("recorded_at") or ""))
    _path(channel_id).write_text(
        json.dumps(rows[-5000:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(videos)
