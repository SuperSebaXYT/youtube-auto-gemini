from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

# Compatibility constants required by generate_kids_video.py
LANDSCAPE = "landscape"
PORTRAIT = "portrait"

# Caption settings
FONT_NAME = "DejaVu Sans"
PORTRAIT_FONT_SIZE = 72
LANDSCAPE_FONT_SIZE = 60
OUTLINE_SIZE = 6

# Captions sit in the lower-middle area, above the YouTube UI
PORTRAIT_MARGIN_V = 420
LANDSCAPE_MARGIN_V = 260

CAPTION_COLORS = [
    "&H00FFFFFF",
    "&H00FFB6FF",
    "&H00B8FFFF",
    "&H00B8FFB8",
    "&H00C8B8FF",
    "&H0080E6FF",
]


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            + " ".join(str(x) for x in cmd)
        )


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read duration from {path}: {result.stderr}"
        )

    return float(result.stdout.strip())


def _build_caption_chunks(
    timing_path: str,
    max_words: int = 2,
) -> List[Tuple[float, float, str]]:
    """
    Reads timing data and groups words into captions.

    Maximum two words per caption.
    Supports common timing formats:
      start|end|word
      start,end,word
      start end word
    """

    if not timing_path or not os.path.exists(timing_path):
        return []

    words = []

    with open(timing_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Ignore obvious headers
            if line.lower().startswith(("start", "word", "time")):
                continue

            parts = re.split(r"[|,\t]+", line)

            if len(parts) < 3:
                parts = line.split()

            if len(parts) < 3:
                continue

            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                continue

            word = " ".join(parts[2:]).strip()

            if not word:
                continue

            words.append((start, end, word))

    if not words:
        return []

    chunks: List[Tuple[float, float, str]] = []

    for i in range(0, len(words), max_words):
        group = words[i:i + max_words]

        start = group[0][0]
        end = group[-1][1]
        text = " ".join(item[2] for item in group)

        chunks.append((start, end, text))

    return chunks


def _ass_time(seconds: float) -> str:
    """
    ASS timestamp format:
    H:MM:SS.cc
    """

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))

    if centiseconds >= 100:
        centiseconds = 0
        secs += 1

    if secs >= 60:
        secs = 0
        minutes += 1

    if minutes >= 60:
        minutes = 0
        hours += 1

    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    """
    Escape characters that have special meaning in ASS.
    """

    text = text.replace("\\", r"\\")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")

    return text


def _write_ass(
    ass_path: str,
    captions: List[Tuple[float, float, str]],
    portrait: bool,
) -> None:

    if portrait:
        font_size = PORTRAIT_FONT_SIZE
        margin_v = PORTRAIT_MARGIN_V
    else:
        font_size = LANDSCAPE_FONT_SIZE
        margin_v = LANDSCAPE_MARGIN_V

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920" if portrait else "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Bubble,{FONT_NAME},{font_size},"
            f"&H00FFFFFF,&H00FFFFFF,&H00101020,&HFF000000,"
            f"1,0,0,0,100,100,0,0,1,{OUTLINE_SIZE},0,"
            f"2,40,40,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for index, (start, end, text) in enumerate(captions):
        if end <= start:
            continue

        color = CAPTION_COLORS[index % len(CAPTION_COLORS)]
        safe_text = _escape_ass_text(text)

        # Large, rounded-looking, thick outlined captions.
        # Slight scale gives a more bubble-like appearance.
        override = (
            f"{{\\c{color}"
            f"\\bord{OUTLINE_SIZE}"
            f"\\shad0"
            f"\\fscx105"
            f"\\fscy105"
            f"\\an2}}"
        )

        line = (
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Bubble,,0,0,0,,"
            f"{override}{safe_text}"
        )

        lines.append(line)

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _normalize_clip(
    input_path: str,
    output_path: str,
    portrait: bool,
) -> None:
    """
    Converts every source clip to a consistent format.
    """

    if portrait:
        scale_filter = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )
    else:
        scale_filter = (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            "crop=1920:1080"
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        scale_filter,
        "-r",
        "30",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    _run(cmd)


def _concat_clips(
    clip_paths: List[str],
    output_path: str,
) -> None:

    concat_file = Path(output_path).with_suffix(".concat.txt")

    with open(concat_file, "w", encoding="utf-8") as f:
        for path in clip_paths:
            escaped = str(Path(path).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        output_path,
    ]

    _run(cmd)

    try:
        concat_file.unlink()
    except OSError:
        pass


def assemble_video(
    clip_paths: List[str],
    audio_path: str,
    timing_path: str,
    output_path: str,
    portrait: bool = True,
) -> str:

    if not clip_paths:
        raise ValueError("No video clips were provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    work_dir = output.parent / "normalized_clips"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_paths: List[str] = []

    print(f"Normalizing {len(clip_paths)} clips...")

    for index, clip in enumerate(clip_paths):
        if not os.path.exists(clip):
            print(f"Skipping missing clip: {clip}")
            continue

        normalized = work_dir / f"clip_{index:03d}.mp4"

        _normalize_clip(
            clip,
            str(normalized),
            portrait=portrait,
        )

        normalized_paths.append(str(normalized))

    if not normalized_paths:
        raise RuntimeError("No valid clips could be normalized.")

    base_video = work_dir / "base_video.mp4"

    print("Concatenating clips...")

    _concat_clips(
        normalized_paths,
        str(base_video),
    )

    # Build captions from timing information.
    captions = _build_caption_chunks(
        timing_path,
        max_words=2,
    )

    ass_path = work_dir / "captions.ass"

    _write_ass(
        str(ass_path),
        captions,
        portrait=portrait,
    )

    # ASS filter needs a properly escaped filesystem path.
    ass_filter_path = str(ass_path.resolve())
    ass_filter_path = ass_filter_path.replace("\\", "/")
    ass_filter_path = ass_filter_path.replace(":", r"\:")

    # Burn captions and add audio.
    final_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(base_video),
        "-i",
        audio_path,
        "-vf",
        f"ass='{ass_filter_path}'",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]

    print("Rendering final video...")

    _run(final_cmd)

    if not output.exists():
        raise RuntimeError(
            f"FFmpeg completed but output file was not created: {output}"
        )

    print(f"Video assembled successfully: {output}")

    return str(output)