from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Any


# ============================================================
# COMPATIBILITY CONSTANTS
# Used by generate_kids_video.py
# ============================================================

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)


# ============================================================
# COSMIC CURIOUS CAPTION SETTINGS
# ============================================================

FONT_NAME = "DejaVu Sans"

PORTRAIT_FONT_SIZE = 72
LANDSCAPE_FONT_SIZE = 60

OUTLINE_SIZE = 6

# Lower-middle of the screen, above YouTube UI
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


# ============================================================
# BASIC FFMPEG HELPERS
# ============================================================

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


def _get_audio_duration(audio_path: str) -> float:
    """
    Returns audio duration in seconds.

    Kept as a public/private helper because
    generate_kids_video.py imports it.
    """

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read audio duration: {result.stderr}"
        )

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid audio duration returned by ffprobe: {result.stdout!r}"
        ) from exc


# ============================================================
# TIMING PARSING
# ============================================================

def _extract_timing_items(data: Any) -> List[Tuple[float, float, str]]:
    """
    Tries to extract word timings from several common JSON formats.
    """

    items: List[Tuple[float, float, str]] = []

    if isinstance(data, dict):
        for key in (
            "words",
            "word_timings",
            "timings",
            "segments",
            "alignment",
        ):
            if key in data:
                return _extract_timing_items(data[key])

        if all(k in data for k in ("start", "end")):
            text = (
                data.get("word")
                or data.get("text")
                or data.get("content")
                or ""
            )

            try:
                items.append(
                    (
                        float(data["start"]),
                        float(data["end"]),
                        str(text).strip(),
                    )
                )
            except (ValueError, TypeError):
                pass

        return items

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                start = (
                    item.get("start")
                    if item.get("start") is not None
                    else item.get("start_time")
                )

                end = (
                    item.get("end")
                    if item.get("end") is not None
                    else item.get("end_time")
                )

                text = (
                    item.get("word")
                    or item.get("text")
                    or item.get("content")
                    or ""
                )

                if start is None or end is None:
                    continue

                try:
                    items.append(
                        (
                            float(start),
                            float(end),
                            str(text).strip(),
                        )
                    )
                except (ValueError, TypeError):
                    continue

            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    items.append(
                        (
                            float(item[0]),
                            float(item[1]),
                            str(item[2]).strip(),
                        )
                    )
                except (ValueError, TypeError):
                    continue

    return items


def _read_timing_file(timing_path: str) -> List[Tuple[float, float, str]]:
    """
    Reads JSON timing data or a simple text timing file.
    """

    if not timing_path or not os.path.exists(timing_path):
        return []

    path = Path(timing_path)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if path.suffix.lower() == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = _extract_timing_items(data)

            if items:
                return [
                    (s, e, t)
                    for s, e, t in items
                    if t and e > s
                ]

        except (json.JSONDecodeError, OSError):
            pass

    # --------------------------------------------------------
    # Text / TSV / CSV / pipe-separated
    # --------------------------------------------------------

    words: List[Tuple[float, float, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.lower().startswith(
                ("start", "word", "time", "begin")
            ):
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

            text = " ".join(parts[2:]).strip()

            if text and end > start:
                words.append((start, end, text))

    return words


def _build_caption_chunks(
    timing_path: str,
    words_per_caption: int = 2,
    max_words: int | None = None,
) -> List[Tuple[float, float, str]]:
    """
    Groups word timings into caption chunks.

    Cosmic Curious uses a maximum of 2 words per caption.

    `words_per_caption` is kept for compatibility with older code.
    `max_words` can also be supplied by newer code.
    """

    if max_words is not None:
        words_per_caption = max_words

    words_per_caption = max(1, int(words_per_caption))

    words = _read_timing_file(timing_path)

    if not words:
        return []

    chunks: List[Tuple[float, float, str]] = []

    for i in range(0, len(words), words_per_caption):
        group = words[i:i + words_per_caption]

        start = group[0][0]
        end = group[-1][1]
        text = " ".join(item[2] for item in group).strip()

        if text and end > start:
            chunks.append((start, end, text))

    return chunks


# ============================================================
# ASS HELPERS
# ============================================================

def _ass_time(seconds: float) -> str:
    """
    Converts seconds to ASS timestamp:
    H:MM:SS.cc
    """

    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    centiseconds = int(
        round((seconds - int(seconds)) * 100)
    )

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
    Escapes ASS special characters.
    """

    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _write_ass_subtitles(
    chunks: List[Tuple[float, float, str]],
    ass_path: Path | str,
    width: int,
    height: int,
    font_size: int,
    portrait: bool = False,
) -> None:
    """
    Compatibility helper used by generate_kids_video.py.

    Creates an ASS subtitle file.
    """

    ass_path = Path(ass_path)

    if portrait:
        margin_v = PORTRAIT_MARGIN_V
    else:
        margin_v = LANDSCAPE_MARGIN_V

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, "
            "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding"
        ),
        (
            f"Style: Default,{FONT_NAME},{font_size},"
            f"&H00FFFFFF,&H00FFFFFF,&H00101020,&HFF000000,"
            f"1,0,0,0,100,100,0,0,1,6,0,2,40,40,"
            f"{margin_v},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, "
            "MarginR, MarginV, Effect, Text"
        ),
    ]

    for start, end, text in chunks:
        if end <= start:
            continue

        safe_text = _escape_ass_text(text)

        lines.append(
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Default,,0,0,0,,"
            f"{{\\bord6\\shad0}}{safe_text}"
        )

    ass_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_ass(
    ass_path: str,
    captions: List[Tuple[float, float, str]],
    portrait: bool,
) -> None:
    """
    ASS writer for Cosmic Curious.
    """

    if portrait:
        font_size = PORTRAIT_FONT_SIZE
        margin_v = PORTRAIT_MARGIN_V
        width = 1080
        height = 1920
    else:
        font_size = LANDSCAPE_FONT_SIZE
        margin_v = LANDSCAPE_MARGIN_V
        width = 1920
        height = 1080

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, "
            "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding"
        ),
        (
            f"Style: Bubble,{FONT_NAME},{font_size},"
            f"&H00FFFFFF,&H00FFFFFF,&H00101020,&HFF000000,"
            f"1,0,0,0,105,105,0,0,1,{OUTLINE_SIZE},0,2,40,40,"
            f"{margin_v},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, MarginL, "
            "MarginR, MarginV, Effect, Text"
        ),
    ]

    for index, (start, end, text) in enumerate(captions):
        if end <= start:
            continue

        color = CAPTION_COLORS[
            index % len(CAPTION_COLORS)
        ]

        safe_text = _escape_ass_text(text)

        override = (
            f"{{\\c{color}"
            f"\\bord{OUTLINE_SIZE}"
            f"\\shad0"
            f"\\fscx105"
            f"\\fscy105"
            f"\\an2}}"
        )

        lines.append(
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Bubble,,0,0,0,,"
            f"{override}{safe_text}"
        )

    Path(ass_path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# VIDEO NORMALIZATION
# ============================================================

def _normalize_clip(
    input_path: str,
    output_path: str,
    portrait: bool,
) -> None:

    if portrait:
        scale_filter = (
            "scale=1080:1920:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920"
        )
    else:
        scale_filter = (
            "scale=1920:1080:"
            "force_original_aspect_ratio=increase,"
            "crop=1920:1080"
        )

    _run([
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
    ])


def _concat_clips(
    clip_paths: List[str],
    output_path: str,
) -> None:

    concat_file = Path(output_path).with_suffix(".concat.txt")

    with open(concat_file, "w", encoding="utf-8") as f:
        for path in clip_paths:
            escaped = (
                str(Path(path).resolve())
                .replace("'", "'\\''")
            )
            f.write(f"file '{escaped}'\n")

    _run([
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
    ])

    try:
        concat_file.unlink()
    except OSError:
        pass


# ============================================================
# MAIN VIDEO ASSEMBLER
# ============================================================

def assemble_video(
    clip_paths: List[str],
    audio_path: str,
    timing_path: str,
    output_path: str,
    portrait: bool = True,
) -> str:

    if not clip_paths:
        raise ValueError("No video clips were provided.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    work_dir = output.parent / "normalized_clips"
    work_dir.mkdir(parents=True, exist_ok=True)

    normalized_paths: List[str] = []

    print(
        f"Normalizing {len(clip_paths)} clips "
        f"({'portrait' if portrait else 'landscape'})..."
    )

    for index, clip in enumerate(clip_paths):
        if not os.path.exists(clip):
            print(f"Skipping missing clip: {clip}")
            continue

        normalized = (
            work_dir / f"clip_{index:03d}.mp4"
        )

        _normalize_clip(
            clip,
            str(normalized),
            portrait,
        )

        normalized_paths.append(str(normalized))

    if not normalized_paths:
        raise RuntimeError(
            "No valid clips could be normalized."
        )

    base_video = work_dir / "base_video.mp4"

    print("Concatenating clips...")

    _concat_clips(
        normalized_paths,
        str(base_video),
    )

    # --------------------------------------------------------
    # Captions
    # --------------------------------------------------------

    captions = _build_caption_chunks(
        timing_path,
        words_per_caption=2,
    )

    ass_path = work_dir / "captions.ass"

    _write_ass(
        str(ass_path),
        captions,
        portrait,
    )

    # --------------------------------------------------------
    # Burn captions + audio
    # --------------------------------------------------------

    ass_filter_path = str(
        ass_path.resolve()
    )

    ass_filter_path = ass_filter_path.replace(
        "\\", "/"
    )

    ass_filter_path = ass_filter_path.replace(
        ":", r"\:"
    )

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
            "FFmpeg finished but output file was not created: "
            f"{output}"
        )

    print(
        f"Video assembled successfully: {output}"
    )

    return str(output)


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_path",
        help="Output video path",
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Audio file",
    )

    parser.add_argument(
        "--timing",
        required=True,
        help="Timing JSON/text file",
    )

    parser.add_argument(
        "--portrait",
        action="store_true",
    )

    parser.add_argument(
        "--clips",
        nargs="+",
        required=True,
        help="Video clips",
    )

    args = parser.parse_args()

    assemble_video(
        args.clips,
        args.audio,
        args.timing,
        args.output_path,
        portrait=args.portrait,
    )