"""
generate_kids_video.py
Builds the 2D cutout-animation video for kids channels (e.g. Jungle Ke
Dost): a set of mascot pose frames (from generate_mascot_assets.py) +
narrated audio + word-timing JSON (from tts.py) -> a video that cuts
between poses on phrase boundaries, so the mascot waves/claps/jumps/dances
along instead of sitting there flapping its mouth.

APPROACH: rather than alpha-overlaying two pixel-aligned frames (the old
mouth-flap system), this concatenates short, independent clips -- one per
pose-hold segment. Because poses are now independently generated images
(no shared base to stay aligned with, see generate_mascot_assets.py's
docstring), a straight cut between them is not just simpler, it's the
correct approach -- trying to alpha-blend unrelated images would look
worse, not better. Discrete pose-swaps are also just how real simple 2D
"cutout style" channels look.

Deliberately reuses assemble_video.py's duration/caption helpers instead
of reimplementing them -- same captions, same libass/HarfBuzz fix for
Hindi text shaping, one place to maintain that logic.

Pose timeline: phrase (caption chunk) boundaries drive pose changes --
"idle" fills silence/gaps between chunks, "talk" is the default while a
chunk is being narrated, and every ACTION_POSE_EVERY-th chunk gets a fun
action pose instead of plain talk, cycling through ACTION_POSES for
variety/rhythm.
"""
import subprocess
import tempfile
from pathlib import Path

from src.assemble_video import (
    LANDSCAPE,
    PORTRAIT,
    _get_audio_duration,
    _build_caption_chunks,
    _write_ass_subtitles,
)

ACTION_POSES = ["wave", "clap", "jump", "dance_left", "dance_right"]
ACTION_POSE_EVERY = 4  # every 4th speaking chunk gets a fun pose instead of plain "talk"
ZOOM_TARGET = 1.06  # gentle Ken Burns zoom per pose-hold segment, subtle so pose cuts read as the main motion


def _make_zoompan_clip(image_path: str, duration: float, width: int, height: int,
                        out_path: Path, fps: int = 30) -> None:
    """Loops a single pose image into a short video with a slow, centered
    zoom-in, so even a held pose doesn't look like a dead static image."""
    frames = max(int(duration * fps), 1)
    zoom_expr = f"min(zoom+{(ZOOM_TARGET - 1) / frames:.8f},{ZOOM_TARGET})"
    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-t", str(duration),
        "-vf", (
            f"scale={width * 2}:{height * 2},"
            f"zoompan=z='{zoom_expr}':d={frames}:s={width}x{height}:fps={fps}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        ),
        "-an", str(out_path),
    ], check=True, capture_output=True)


def _build_pose_timeline(chunks: list, audio_duration: float) -> list:
    """Returns [(start, end, pose_name), ...] covering [0, audio_duration)
    with no gaps -- every caption chunk becomes a "talk" (or, periodically,
    an action pose) segment, and every gap between/around chunks becomes
    an "idle" segment."""
    if not chunks:
        return [(0.0, audio_duration, "idle")]

    timeline = []
    cursor = 0.0
    action_i = 0
    for i, (start, end, _text) in enumerate(chunks):
        if start > cursor:
            timeline.append((cursor, start, "idle"))
        pose = "talk"
        if (i + 1) % ACTION_POSE_EVERY == 0:
            pose = ACTION_POSES[action_i % len(ACTION_POSES)]
            action_i += 1
        timeline.append((start, end, pose))
        cursor = end
    if cursor < audio_duration:
        timeline.append((cursor, audio_duration, "idle"))
    return timeline


def generate_kids_video(
    audio_path: str,
    timing_path: str,
    frame_paths: dict,
    output_path: str,
    portrait: bool = False,
) -> None:
    """frame_paths is {pose_name: path} as returned by
    generate_mascot_assets() -- one entry per pose in that module's
    POSES list."""
    width, height = PORTRAIT if portrait else LANDSCAPE
    audio_duration = _get_audio_duration(audio_path)
    chunks = _build_caption_chunks(timing_path)
    timeline = _build_pose_timeline(chunks, audio_duration)
    idle_frame = frame_paths.get("idle", next(iter(frame_paths.values())))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. One short zoompan clip per pose-hold segment.
        segment_clips = []
        for i, (start, end, pose) in enumerate(timeline):
            duration = max(end - start, 0.05)  # guard against a zero-length segment
            image_path = frame_paths.get(pose, idle_frame)
            clip_path = tmp / f"segment_{i:04d}.mp4"
            _make_zoompan_clip(image_path, duration, width, height, clip_path)
            segment_clips.append(clip_path)

        # 2. Concatenate into the full-duration background track.
        concat_list = tmp / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p}'" for p in segment_clips))
        background = tmp / "background.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(background),
        ], check=True, capture_output=True)

        # 3. Captions -- identical helpers/settings to assemble_video.py.
        font_size = 56 if portrait else 44
        vf_chain = None
        if chunks:
            ass_path = tmp / "captions.ass"
            _write_ass_subtitles(chunks, ass_path, width, height, font_size, portrait)
            escaped_ass_path = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
            font_dir = str(Path.home() / ".local" / "share" / "fonts").replace(":", r"\:")
            vf_chain = f"subtitles='{escaped_ass_path}':fontsdir='{font_dir}'"

        # 4. Burn captions on top (if any) and mux the narration audio in.
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", str(background), "-i", audio_path]
        if vf_chain:
            cmd += ["-vf", vf_chain]
        cmd += [
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument("timing_path")
    parser.add_argument("output_path")
    parser.add_argument("frame_paths_json", help='JSON object, e.g. \'{"idle": "idle.png", "talk": "talk.png", ...}\'')
    parser.add_argument("--portrait", action="store_true")
    args = parser.parse_args()

    generate_kids_video(args.audio_path, args.timing_path,
                         json.loads(args.frame_paths_json),
                         args.output_path, args.portrait)
    print(f"Wrote {args.output_path}")
