from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Any

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)

FONT_NAME = "DejaVu Sans"

PORTRAIT_FONT_SIZE = 96
LANDSCAPE_FONT_SIZE = 76

MIN_PORTRAIT_FONT_SIZE = 58
MIN_LANDSCAPE_FONT_SIZE = 48

OUTLINE_SIZE = 9
CAPTION_MAX_WIDTH_RATIO = 0.84

PORTRAIT_MARGIN_V = 430
LANDSCAPE_MARGIN_V = 270

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


def _get_audio_duration(audio_path: str) -> float:
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
            f"Invalid audio duration returned by ffprobe: "
            f"{result.stdout!r}"
        ) from exc


def _extract_timing_items(
    data: Any,
) -> List[Tuple[float, float, str]]:
    """
    Convert timing data into:

        (start_seconds, end_seconds, text)

    Supports both:

    1. start/end
    2. start_time/end_time
    3. Edge-TTS format used by tts.py:
       offset_seconds/duration_seconds
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
                return _extract_timing_items(
                    data[key]
                )

        # ---------------------------------------------------------
        # Standard start/end format
        # ---------------------------------------------------------

        if all(
            k in data
            for k in ("start", "end")
        ):
            text = (
                data.get("word")
                or data.get("text")
                or data.get("content")
                or ""
            )

            try:
                start = float(data["start"])
                end = float(data["end"])

                if (
                    start >= 0
                    and end > start
                    and str(text).strip()
                ):
                    items.append(
                        (
                            start,
                            end,
                            str(text).strip(),
                        )
                    )

            except (
                ValueError,
                TypeError,
            ):
                pass

        # ---------------------------------------------------------
        # Edge-TTS format:
        #
        # offset_seconds
        # duration_seconds
        # ---------------------------------------------------------

        elif all(
            k in data
            for k in (
                "offset_seconds",
                "duration_seconds",
            )
        ):
            text = (
                data.get("text")
                or data.get("word")
                or data.get("content")
                or ""
            )

            try:
                start = float(
                    data["offset_seconds"]
                )

                duration = float(
                    data["duration_seconds"]
                )

                end = start + duration

                if (
                    start >= 0
                    and duration > 0
                    and end > start
                    and str(text).strip()
                ):
                    items.append(
                        (
                            start,
                            end,
                            str(text).strip(),
                        )
                    )

            except (
                ValueError,
                TypeError,
            ):
                pass

        return items

    if isinstance(data, list):

        for item in data:

            if isinstance(item, dict):

                # -------------------------------------------------
                # Standard start/end
                # -------------------------------------------------

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

                # -------------------------------------------------
                # Edge-TTS format
                # -------------------------------------------------

                if (
                    start is None
                    and end is None
                    and item.get(
                        "offset_seconds"
                    ) is not None
                    and item.get(
                        "duration_seconds"
                    ) is not None
                ):
                    try:
                        start = float(
                            item[
                                "offset_seconds"
                            ]
                        )

                        duration = float(
                            item[
                                "duration_seconds"
                            ]
                        )

                        end = start + duration

                    except (
                        ValueError,
                        TypeError,
                    ):
                        continue

                if (
                    start is None
                    or end is None
                ):
                    continue

                try:
                    start = float(start)
                    end = float(end)

                    cleaned_text = str(
                        text
                    ).strip()

                    if (
                        start >= 0
                        and end > start
                        and cleaned_text
                    ):
                        items.append(
                            (
                                start,
                                end,
                                cleaned_text,
                            )
                        )

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

            elif (
                isinstance(
                    item,
                    (list, tuple),
                )
                and len(item) >= 3
            ):
                try:
                    start = float(item[0])
                    end = float(item[1])
                    text = str(
                        item[2]
                    ).strip()

                    if (
                        start >= 0
                        and end > start
                        and text
                    ):
                        items.append(
                            (
                                start,
                                end,
                                text,
                            )
                        )

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

    return items


def _read_timing_file(
    timing_path: str,
) -> List[Tuple[float, float, str]]:

    if not timing_path or not os.path.exists(
        timing_path
    ):
        print(
            f"[CAPTIONS] Timing file not found: "
            f"{timing_path}"
        )
        return []

    path = Path(timing_path)

    if path.suffix.lower() == ".json":

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            items = _extract_timing_items(
                data
            )

            cleaned = [
                (s, e, t)
                for s, e, t in items
                if t and e > s
            ]

            if cleaned:
                print(
                    f"[CAPTIONS] Read "
                    f"{len(cleaned)} word timings"
                )
                return cleaned

        except (
            json.JSONDecodeError,
            OSError,
        ) as exc:
            print(
                f"[CAPTIONS] Could not read JSON "
                f"timing file: {exc}"
            )

    words: List[
        Tuple[float, float, str]
    ] = []

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            for raw_line in f:

                line = raw_line.strip()

                if not line:
                    continue

                if line.lower().startswith(
                    (
                        "start",
                        "word",
                        "time",
                        "begin",
                    )
                ):
                    continue

                parts = re.split(
                    r"[|,\t]+",
                    line,
                )

                if len(parts) < 3:
                    parts = line.split()

                if len(parts) < 3:
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])

                except ValueError:
                    continue

                text = " ".join(
                    parts[2:]
                ).strip()

                if (
                    text
                    and end > start
                ):
                    words.append(
                        (
                            start,
                            end,
                            text,
                        )
                    )

    except OSError as exc:
        print(
            f"[CAPTIONS] Could not read timing "
            f"file: {exc}"
        )
        return []

    print(
        f"[CAPTIONS] Read "
        f"{len(words)} word timings"
    )

    return words


def _build_caption_chunks(
    timing_path: str,
    words_per_caption: int = 2,
    max_words: int | None = None,
) -> List[Tuple[float, float, str]]:

    if max_words is not None:
        words_per_caption = max_words

    words_per_caption = max(
        1,
        int(words_per_caption),
    )

    words = _read_timing_file(
        timing_path
    )

    if not words:
        print(
            "[CAPTIONS] No usable timings."
        )
        return []

    chunks: List[
        Tuple[float, float, str]
    ] = []

    for i in range(
        0,
        len(words),
        words_per_caption,
    ):

        group = words[
            i:i + words_per_caption
        ]

        start = group[0][0]
        end = group[-1][1]

        text = " ".join(
            item[2]
            for item in group
        ).strip()

        if (
            text
            and end > start
        ):
            chunks.append(
                (
                    start,
                    end,
                    text,
                )
            )

    print(
        f"[CAPTIONS] Created "
        f"{len(chunks)} captions "
        f"with max "
        f"{words_per_caption} words"
    )

    return chunks


def _ass_time(seconds: float) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    centiseconds = int(
        round(
            (
                seconds
                - int(seconds)
            ) * 100
        )
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

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:02d}."
        f"{centiseconds:02d}"
    )


def _escape_ass_text(
    text: str,
) -> str:

    return (
        str(text)
        .replace(
            "\\",
            r"\\",
        )
        .replace(
            "{",
            r"\{",
        )
        .replace(
            "}",
            r"\}",
        )
    )


def _estimate_text_width(
    text: str,
    font_size: int,
) -> float:

    words = text.split()

    if not words:
        return 0.0

    total = 0.0

    for word in words:

        for char in word:

            if char in "ilI.,'!|":
                factor = 0.28

            elif char in "mwMW@#":
                factor = 0.90

            elif char.isupper():
                factor = 0.68

            elif char.isdigit():
                factor = 0.62

            else:
                factor = 0.56

            total += (
                font_size * factor
            )

        total += (
            font_size * 0.20
        )

    return total


def _caption_font_size(
    text: str,
    portrait: bool,
) -> int:

    if portrait:
        max_size = PORTRAIT_FONT_SIZE
        min_size = MIN_PORTRAIT_FONT_SIZE
        width = PORTRAIT[0]

    else:
        max_size = LANDSCAPE_FONT_SIZE
        min_size = MIN_LANDSCAPE_FONT_SIZE
        width = LANDSCAPE[0]

    max_width = (
        width
        * CAPTION_MAX_WIDTH_RATIO
    )

    size = max_size

    while size > min_size:

        estimated_width = (
            _estimate_text_width(
                text,
                size,
            )
        )

        if estimated_width <= max_width:
            break

        size -= 2

    return max(
        min_size,
        size,
    )


def _write_ass_subtitles(
    chunks: List[
        Tuple[float, float, str]
    ],
    ass_path: Path | str,
    width: int,
    height: int,
    font_size: int,
    portrait: bool = False,
) -> None:

    margin_v = (
        PORTRAIT_MARGIN_V
        if portrait
        else LANDSCAPE_MARGIN_V
    )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, "
            "PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding"
        ),
        (
            f"Style: Default,{FONT_NAME},{font_size},"
            f"&H00FFFFFF,&H00FFFFFF,"
            f"&H00000000,&H00000000,"
            f"1,0,0,0,100,100,0,0,1,"
            f"6,0,2,40,40,{margin_v},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, "
            "MarginL, MarginR, MarginV, Effect, Text"
        ),
    ]

    for start, end, text in chunks:

        if end <= start:
            continue

        safe_text = _escape_ass_text(
            text
        )

        lines.append(
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Default,,0,0,0,,"
            f"{{\\bord6\\shad0}}"
            f"{safe_text}"
        )

    Path(ass_path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_ass(
    ass_path: str,
    captions: List[
        Tuple[float, float, str]
    ],
    portrait: bool,
) -> None:

    if portrait:
        default_font_size = (
            PORTRAIT_FONT_SIZE
        )
        margin_v = PORTRAIT_MARGIN_V
        width = 1080
        height = 1920

    else:
        default_font_size = (
            LANDSCAPE_FONT_SIZE
        )
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
            "Format: Name, Fontname, Fontsize, "
            "PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, "
            "Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, "
            "MarginV, Encoding"
        ),
        (
            f"Style: Bubble,{FONT_NAME},"
            f"{default_font_size},"
            f"&H00FFFFFF,&H00FFFFFF,"
            f"&H00000000,&H00000000,"
            f"1,0,0,0,105,105,0,0,1,"
            f"{OUTLINE_SIZE},0,2,40,40,"
            f"{margin_v},1"
        ),
        "",
        "[Events]",
        (
            "Format: Layer, Start, End, Style, Name, "
            "MarginL, MarginR, MarginV, Effect, Text"
        ),
    ]

    for index, (
        start,
        end,
        text,
    ) in enumerate(captions):

        if end <= start:
            continue

        font_size = _caption_font_size(
            text,
            portrait,
        )

        color = CAPTION_COLORS[
            index % len(
                CAPTION_COLORS
            )
        ]

        safe_text = _escape_ass_text(
            text
        )

        override = (
            f"{{"
            f"\\c{color}"
            f"\\fs{font_size}"
            f"\\bord{OUTLINE_SIZE}"
            f"\\shad0"
            f"\\fscx105"
            f"\\fscy105"
            f"\\an2"
            f"}}"
        )

        lines.append(
            f"Dialogue: 0,"
            f"{_ass_time(start)},"
            f"{_ass_time(end)},"
            f"Bubble,,0,0,0,,"
            f"{override}"
            f"{safe_text}"
        )

    Path(ass_path).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"[CAPTIONS] ASS created: "
        f"{len(captions)} captions"
    )


def _normalize_clip(
    input_path: str,
    output_path: str,
    portrait: bool,
    duration: float | None = None,
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

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
    ]

    if duration is not None:

        cmd.extend([
            "-t",
            f"{duration:.3f}",
        ])

    cmd.extend([
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

    _run(cmd)


def _concat_clips(
    clip_paths: List[str],
    output_path: str,
) -> None:

    concat_file = Path(
        output_path
    ).with_suffix(
        ".concat.txt"
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8",
    ) as f:

        for path in clip_paths:

            escaped = (
                str(
                    Path(path).resolve()
                )
                .replace(
                    "'",
                    "'\\''",
                )
            )

            f.write(
                f"file '{escaped}'\n"
            )

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


def assemble_video(
    clip_paths: List[str],
    audio_path: str,
    timing_path: str,
    output_path: str,
    portrait: bool = True,
) -> str:

    if not clip_paths:
        raise ValueError(
            "No video clips were provided."
        )

    if not os.path.exists(
        audio_path
    ):
        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir = (
        output.parent
        / "normalized_clips"
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_duration = (
        _get_audio_duration(
            audio_path
        )
    )

    print(
        f"[VIDEO] Audio duration: "
        f"{audio_duration:.2f}s"
    )

    clip_count = len(
        clip_paths
    )

    if portrait and clip_count > 1:
        segment_duration = (
            audio_duration
            / clip_count
        )

    else:
        segment_duration = None

    print(
        f"[VIDEO] Source clips: "
        f"{clip_count}"
    )

    if segment_duration:
        print(
            f"[VIDEO] Target visual duration: "
            f"{segment_duration:.2f}s"
        )

    normalized_paths: List[
        str
    ] = []

    print(
        f"[VIDEO] Normalizing "
        f"{clip_count} clips..."
    )

    for index, clip in enumerate(
        clip_paths
    ):

        if not os.path.exists(
            clip
        ):

            print(
                f"[VIDEO] Skipping missing clip: "
                f"{clip}"
            )

            continue

        normalized = (
            work_dir
            / f"clip_{index:03d}.mp4"
        )

        _normalize_clip(
            clip,
            str(normalized),
            portrait,
            duration=segment_duration,
        )

        normalized_paths.append(
            str(normalized)
        )

    if not normalized_paths:
        raise RuntimeError(
            "No valid clips could be normalized."
        )

    base_video = (
        work_dir
        / "base_video.mp4"
    )

    print(
        f"[VIDEO] Concatenating "
        f"{len(normalized_paths)} "
        f"visual segments..."
    )

    _concat_clips(
        normalized_paths,
        str(base_video),
    )

    captions = (
        _build_caption_chunks(
            timing_path,
            words_per_caption=2,
        )
    )

    ass_path = (
        work_dir
        / "captions.ass"
    )

    _write_ass(
        str(ass_path),
        captions,
        portrait,
    )

    ass_filter_path = str(
        ass_path.resolve()
    )

    ass_filter_path = (
        ass_filter_path
        .replace(
            "\\",
            "/",
        )
    )

    ass_filter_path = (
        ass_filter_path
        .replace(
            ":",
            r"\:",
        )
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

    print(
        "[VIDEO] Rendering final video..."
    )

    _run(
        final_cmd
    )

    if not output.exists():
        raise RuntimeError(
            "FFmpeg finished but output "
            "file was not created: "
            f"{output}"
        )

    print(
        f"Video assembled successfully: "
        f"{output}"
    )

    return str(output)


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