from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Tuple


# ============================================================
# CONFIG
# ============================================================

PORTRAIT_WIDTH = 1080
PORTRAIT_HEIGHT = 1920

LANDSCAPE_WIDTH = 1920
LANDSCAPE_HEIGHT = 1080

FPS = 30

# Duża, zaokrąglona czcionka.
# Jeśli później dodamy konkretny bubble font do repo,
# wystarczy zmienić tę nazwę.
FONT_NAME = "DejaVu Sans"

PORTRAIT_FONT_SIZE = 72
LANDSCAPE_FONT_SIZE = 60

# Gruby obrys
OUTLINE_SIZE = 6

# Pozycja napisów:
# Alignment=2 = dół-środek.
# MarginV określa odległość od dołu.
#
# Dla Shortów tekst będzie więc lekko nad dolną częścią
# ekranu, mniej więcej w miejscu nad ikonkami/serduszkiem.
PORTRAIT_MARGIN_V = 420
LANDSCAPE_MARGIN_V = 260


# ============================================================
# HELPERS
# ============================================================

def _run(cmd: List[str]) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\n"
            + result.stderr[-4000:]
        )


def _probe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not read duration of {path}:\n{result.stderr}"
        )

    return float(result.stdout.strip())


# ============================================================
# CAPTIONS
# ============================================================

def _build_caption_chunks(
    timing_path: str,
    max_words: int = 2,
) -> List[Tuple[float, float, str]]:
    """
    Reads word timing JSON and creates caption chunks.

    IMPORTANT:
    Maximum 2 words per caption.
    """

    with open(timing_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support several possible timing JSON structures.
    if isinstance(data, dict):
        words = data.get("words", data.get("timings", []))
    else:
        words = data

    if not words:
        return []

    chunks: List[Tuple[float, float, str]] = []

    current_words: List[str] = []
    start_time = None
    end_time = None

    for item in words:
        if not isinstance(item, dict):
            continue

        word = str(
            item.get("word")
            or item.get("text")
            or ""
        ).strip()

        if not word:
            continue

        start = item.get("start")
        end = item.get("end")

        if start is None:
            start = item.get("start_time")

        if end is None:
            end = item.get("end_time")

        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)

        if start_time is None:
            start_time = start

        current_words.append(word)
        end_time = end

        if len(current_words) >= max_words:
            chunks.append(
                (
                    start_time,
                    end_time,
                    " ".join(current_words),
                )
            )

            current_words = []
            start_time = None
            end_time = None

    if current_words and start_time is not None and end_time is not None:
        chunks.append(
            (
                start_time,
                end_time,
                " ".join(current_words),
            )
        )

    return chunks


def _ass_time(seconds: float) -> str:
    """
    ASS timestamp:
    H:MM:SS.cc
    """

    total_cs = max(0, int(round(seconds * 100)))

    hours = total_cs // 360000
    total_cs %= 360000

    minutes = total_cs // 6000
    total_cs %= 6000

    secs = total_cs // 100
    centiseconds = total_cs % 100

    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _write_ass_subtitles(
    chunks: List[Tuple[float, float, str]],
    output_path: str,
    portrait: bool,
) -> None:
    """
    Creates ASS subtitles with:
    - large rounded-looking font
    - thick outline
    - no rectangular background
    - bottom-center positioning
    - max 2 words per caption
    """

    if portrait:
        font_size = PORTRAIT_FONT_SIZE
        margin_v = PORTRAIT_MARGIN_V
    else:
        font_size = LANDSCAPE_FONT_SIZE
        margin_v = LANDSCAPE_MARGIN_V

    # ASS colors use AABBGGRR.
    #
    # Main color: bright white.
    # Outline: dark purple/black.
    #
    # The background is fully transparent.
    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PORTRAIT_WIDTH if portrait else LANDSCAPE_WIDTH}
PlayResY: {PORTRAIT_HEIGHT if portrait else LANDSCAPE_HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Bubble,{FONT_NAME},{font_size},&H00FFFFFF,&H00FFFFFF,&H00101020,&HFF000000,1,0,0,0,100,100,0,0,1,{OUTLINE_SIZE},0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Kolory głównego tekstu.
    # Każdy kolejny napis może dostać inny kolor.
    colors = [
        "&H00FFFFFF",  # white
        "&H00FFB6FF",  # pink
        "&H00B8FFFF",  # cyan
        "&H00B8FFB8",  # green
        "&H00C8B8FF",  # purple
        "&H0080E6FF",  # yellow/orange-ish
    ]

    for index, (start, end, text) in enumerate(chunks):
        # Escape znaków specjalnych ASS.
        safe_text = (
            text
            .replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
        )

        color = colors[index % len(colors)]

        # Lekki efekt "pop":
        # tekst jest normalnie wyświetlany,
        # ale pojawia się z minimalnym przesunięciem/skalami.
        #
        # Bez przesadnego trzęsienia, żeby napis pozostał czytelny.
        dialogue_text = (
            f"{{\\c{color}}}"
            f"{{\\bord{OUTLINE_SIZE}}}"
            f"{{\\shad0}}"
            f"{{\\an2}}"
            f"{safe_text}"
        )

        ass += (
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Bubble,"
            f",0,0,0,,"
            f"{dialogue_text}\n"
        )

    Path(output_path).write_text(
        ass,
        encoding="utf-8",
    )


# ============================================================
# VIDEO NORMALIZATION
# ============================================================

def _normalize_clip(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    duration: float,
) -> None:
    """
    Converts stock footage to the target resolution/FPS.
    """

    vf = (
        f"scale={width}:{height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={FPS},"
        f"setsar=1"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-t",
        str(max(duration, 0.1)),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    _run(cmd)


# ============================================================
# MAIN ASSEMBLY
# ============================================================

def assemble_video(
    clip_paths: List[str],
    audio_path: str,
    timing_path: str,
    output_path: str,
    portrait: bool = True,
) -> None:
    """
    Assemble stock clips + audio + captions.

    For Shorts:
        1080x1920
        large captions
        max 2 words
        captions near lower-center

    For landscape:
        1920x1080
    """

    if not clip_paths:
        raise ValueError("No clip paths provided.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # AUDIO DURATION
    # --------------------------------------------------------

    audio_duration = _probe_duration(audio_path)

    if audio_duration <= 0:
        raise ValueError("Audio duration is invalid.")

    # --------------------------------------------------------
    # TARGET RESOLUTION
    # --------------------------------------------------------

    if portrait:
        width = PORTRAIT_WIDTH
        height = PORTRAIT_HEIGHT
    else:
        width = LANDSCAPE_WIDTH
        height = LANDSCAPE_HEIGHT

    # --------------------------------------------------------
    # TEMP DIRECTORY
    # --------------------------------------------------------

    temp_dir = output.parent / "_assembly_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    normalized_paths: List[str] = []

    # Each clip gets an equal portion of the audio duration.
    clip_duration = audio_duration / len(clip_paths)

    # --------------------------------------------------------
    # NORMALIZE CLIPS
    # --------------------------------------------------------

    for index, clip_path in enumerate(clip_paths):
        normalized = temp_dir / f"clip_{index:03d}.mp4"

        _normalize_clip(
            clip_path,
            str(normalized),
            width,
            height,
            clip_duration,
        )

        normalized_paths.append(str(normalized))

    # --------------------------------------------------------
    # CONCAT LIST
    # --------------------------------------------------------

    concat_file = temp_dir / "concat.txt"

    with open(concat_file, "w", encoding="utf-8") as f:
        for path in normalized_paths:
            escaped = path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    concatenated = temp_dir / "concatenated.mp4"

    cmd_concat = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-t",
        str(audio_duration),
        "-c",
        "copy",
        str(concatenated),
    ]

    _run(cmd_concat)

    # --------------------------------------------------------
    # CAPTIONS
    # --------------------------------------------------------

    caption_chunks = _build_caption_chunks(
        timing_path,
        max_words=2,
    )

    ass_path = temp_dir / "captions.ass"

    _write_ass_subtitles(
        caption_chunks,
        str(ass_path),
        portrait=portrait,
    )

    # --------------------------------------------------------
    # FINAL VIDEO
    # --------------------------------------------------------

    # Escape path for FFmpeg subtitles filter.
    ass_filter_path = str(ass_path).replace("\\", "/")
    ass_filter_path = ass_filter_path.replace(":", "\\:")

    video_filter = f"ass='{ass_filter_path}'"

    cmd_final = [
        "ffmpeg",
        "-y",
        "-i",
        str(concatenated),
        "-i",
        audio_path,
        "-vf",
        video_filter,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        str(audio_duration),
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
        str(output),
    ]

    _run(cmd_final)

    # --------------------------------------------------------
    # CLEAN TEMP FILES
    # --------------------------------------------------------

    for path in normalized_paths:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass

    for path in [concat_file, concatenated, ass_path]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    try:
        temp_dir.rmdir()
    except OSError:
        pass