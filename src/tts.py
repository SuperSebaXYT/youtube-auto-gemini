
"""
tts.py

Converts script text to speech using edge-tts.

Produces word-level timing data for:
    - synced captions
    - mascot mouth-flap timing
    - animation timing

Timing strategy:
1. Prefer native WordBoundary events.
2. Validate native timing data.
3. Estimate timings from actual audio duration if native
   timing data is unavailable or unreliable.

The timing JSON remains a PLAIN LIST for compatibility.
"""

import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts
from aiohttp.client_exceptions import WSServerHandshakeError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_VALID_BOUNDARIES = 3
MIN_BOUNDARY_COVERAGE = 0.70

TTS_MAX_ATTEMPTS = 3
TTS_RETRY_DELAY_SECONDS = 2


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _split_words(text: str) -> list:
    return [word for word in text.split() if word.strip()]


def _clean_word_for_weight(word: str) -> str:
    cleaned = re.sub(r"[^\w'-]", "", word, flags=re.UNICODE)
    return cleaned or word


def _count_words(text: str) -> int:
    return len(_split_words(text))


# ---------------------------------------------------------------------------
# Edge-TTS synthesis with retries
# ---------------------------------------------------------------------------

async def _synthesize(
    text: str,
    voice: str,
    audio_out: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> list:
    """
    Generate audio and collect native WordBoundary events.

    Retries temporary failures up to TTS_MAX_ATTEMPTS times.
    Each attempt writes to a temporary file. The final audio file is
    replaced only after the stream completes successfully.
    """

    last_error = None

    for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
        word_boundaries = []

        temp_out = audio_out.with_name(
            f"{audio_out.stem}.attempt{attempt}.mp3"
        )

        try:
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=rate,
                pitch=pitch,
            )

            with open(temp_out, "wb") as audio_file:
                async for chunk in communicate.stream():
                    chunk_type = chunk.get("type")

                    if chunk_type == "audio":
                        data = chunk.get("data")
                        if data:
                            audio_file.write(data)

                    elif chunk_type == "WordBoundary":
                        try:
                            word = str(
                                chunk.get("text", "")
                            ).strip()

                            if not word:
                                continue

                            offset_seconds = (
                                float(chunk["offset"]) / 10_000_000
                            )

                            duration_seconds = (
                                float(chunk["duration"]) / 10_000_000
                            )

                            if offset_seconds < 0:
                                continue

                            if duration_seconds <= 0:
                                continue

                            word_boundaries.append({
                                "text": word,
                                "offset_seconds": offset_seconds,
                                "duration_seconds": duration_seconds,
                            })

                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                        ):
                            continue

            if (
                not temp_out.exists()
                or temp_out.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Edge TTS returned no audio data."
                )

            # Replace the final file only after successful completion.
            temp_out.replace(audio_out)

            print(
                f"TTS: synthesis succeeded on attempt "
                f"{attempt}/{TTS_MAX_ATTEMPTS}."
            )

            return word_boundaries

        except Exception as exc:
            last_error = exc

            if temp_out.exists():
                temp_out.unlink()

            print(
                f"WARNING: TTS attempt "
                f"{attempt}/{TTS_MAX_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < TTS_MAX_ATTEMPTS:
                delay = TTS_RETRY_DELAY_SECONDS * attempt
                print(f"TTS: retrying in {delay} seconds...")
                await asyncio.sleep(delay)

    if isinstance(last_error, WSServerHandshakeError):
        raise RuntimeError(
            "Edge TTS rejected the connection after "
            f"{TTS_MAX_ATTEMPTS} attempts. "
            "The server returned an HTTP WebSocket handshake error, "
            "such as 403. Check the installed edge-tts version and "
            "service availability. If the error persists, use a "
            "separate TTS provider."
        ) from last_error

    raise RuntimeError(
        f"Edge TTS failed after {TTS_MAX_ATTEMPTS} attempts: "
        f"{last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Audio duration
# ---------------------------------------------------------------------------

def _get_audio_duration(audio_path: Path) -> float:
    """Get the actual generated audio duration using ffprobe."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    value = result.stdout.strip()

    if not value:
        raise RuntimeError(
            f"ffprobe returned no duration for {audio_path}"
        )

    duration = float(value)

    if duration <= 0:
        raise RuntimeError(
            f"Invalid audio duration: {duration}"
        )

    return duration


# ---------------------------------------------------------------------------
# Native timing validation
# ---------------------------------------------------------------------------

def _native_timings_are_valid(
    text: str,
    timings: list,
    audio_duration: float,
) -> bool:
    """Check whether native WordBoundary metadata is sufficiently complete."""

    if not timings:
        return False

    if len(timings) < MIN_VALID_BOUNDARIES:
        return False

    expected_words = _count_words(text)

    if expected_words <= 0:
        return False

    coverage = len(timings) / expected_words

    if coverage < MIN_BOUNDARY_COVERAGE:
        return False

    previous_end = 0.0
    valid_entries = 0

    for item in timings:
        try:
            start = float(item["offset_seconds"])
            duration = float(item["duration_seconds"])
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        end = start + duration

        if start < 0 or duration <= 0:
            continue

        if start > audio_duration + 0.5:
            continue

        if end > audio_duration + 1.0:
            continue

        if start + 0.001 < previous_end:
            continue

        previous_end = end
        valid_entries += 1

    return valid_entries >= MIN_VALID_BOUNDARIES


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_into_timing_segments(text: str) -> list:
    """Split narration into natural sentence segments."""

    segments = re.findall(
        r"[^.!?]+(?:[.!?]+|$)",
        text,
        flags=re.UNICODE,
    )

    segments = [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]

    if not segments:
        return [text.strip()]

    return segments


# ---------------------------------------------------------------------------
# Word timing weights
# ---------------------------------------------------------------------------

def _sentence_weights(sentence: str) -> list:
    """Estimate relative word durations using word length and punctuation."""

    words = _split_words(sentence)

    if not words:
        return []

    weights = []

    for word in words:
        clean = _clean_word_for_weight(word)

        weight = max(
            1.0,
            len(clean) ** 0.70,
        )

        if word.endswith((",", ";", ":")):
            weight *= 1.08

        if word.endswith((".", "!", "?")):
            weight *= 1.15

        weights.append(weight)

    return weights


# ---------------------------------------------------------------------------
# Estimated word timings
# ---------------------------------------------------------------------------

def _estimate_word_timings(
    text: str,
    audio_out: Path,
) -> list:
    """
    Estimate word timings from actual audio duration.

    This is an approximation. It does not replace genuine forced alignment.
    """

    words = _split_words(text)

    if not words:
        return []

    duration = _get_audio_duration(audio_out)

    if duration <= 0:
        return []

    sentences = _split_into_timing_segments(text)

    sentence_word_counts = [
        len(_split_words(sentence))
        for sentence in sentences
    ]

    total_words = sum(sentence_word_counts)

    if total_words <= 0:
        return []

    sentence_weights = []

    for sentence, count in zip(
        sentences,
        sentence_word_counts,
    ):
        punctuation_bonus = 1.0

        if sentence.rstrip().endswith((".", "!", "?")):
            punctuation_bonus = 1.10

        sentence_weights.append(
            max(1.0, count * punctuation_bonus)
        )

    total_sentence_weight = sum(sentence_weights)

    timings = []
    global_offset = 0.0

    for sentence, sentence_weight in zip(
        sentences,
        sentence_weights,
    ):
        sentence_words = _split_words(sentence)

        if not sentence_words:
            continue

        sentence_duration = (
            duration
            * sentence_weight
            / total_sentence_weight
        )

        weights = _sentence_weights(sentence)

        total_weight = sum(weights) or len(sentence_words)

        local_offset = global_offset

        for word, weight in zip(sentence_words, weights):
            word_duration = (
                sentence_duration
                * weight
                / total_weight
            )

            timings.append({
                "text": word,
                "offset_seconds": local_offset,
                "duration_seconds": word_duration,
            })

            local_offset += word_duration

        global_offset += sentence_duration

    # Correct floating-point drift so the final word ends at audio duration.
    if timings:
        final = timings[-1]

        final_end = (
            final["offset_seconds"]
            + final["duration_seconds"]
        )

        correction = duration - final_end

        if correction > 0:
            final["duration_seconds"] += correction

    return timings


# ---------------------------------------------------------------------------
# Main synthesis function
# ---------------------------------------------------------------------------

def synthesize_speech(
    text: str,
    voice: str,
    audio_out: str,
    timing_out: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> None:
    """
    Generate speech and word timing data.

    timing_out remains a JSON LIST for compatibility with the video pipeline.
    """

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    if not voice or not voice.strip():
        raise ValueError("TTS voice is empty.")

    audio_out = Path(audio_out)
    timing_out = Path(timing_out)

    audio_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timing_out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale output so an earlier run cannot be mistaken for success.
    if audio_out.exists():
        audio_out.unlink()

    word_boundaries = asyncio.run(
        _synthesize(
            text=text,
            voice=voice,
            audio_out=audio_out,
            rate=rate,
            pitch=pitch,
        )
    )

    audio_duration = _get_audio_duration(audio_out)

    if _native_timings_are_valid(
        text=text,
        timings=word_boundaries,
        audio_duration=audio_duration,
    ):
        timings = word_boundaries

        print(
            f"TTS: voice '{voice}' returned "
            f"{len(timings)} reliable native WordBoundary events."
        )

        print("TTS: using native word timings.")

    else:
        print(
            f"WARNING: voice '{voice}' returned no reliable "
            "WordBoundary events from edge-tts."
        )

        print(
            "TTS: estimating word timings from actual audio duration."
        )

        timings = _estimate_word_timings(
            text=text,
            audio_out=audio_out,
        )

        print(
            f"TTS: generated {len(timings)} estimated word timings."
        )

        print(
            f"TTS: actual audio duration = {audio_duration:.2f}s"
        )

    if not timings:
        raise RuntimeError(
            "TTS produced audio but no usable word timings could be generated."
        )

    # Preserve the plain-list JSON format expected by existing pipeline code.
    timing_out.write_text(
        json.dumps(
            timings,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"TTS: wrote audio: {audio_out}")
    print(f"TTS: wrote timing: {timing_out}")
    print(f"TTS: timing entries: {len(timings)}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Edge-TTS narration and word timings."
    )

    parser.add_argument(
        "text_file",
        help="Path to a .txt file containing the script",
    )

    parser.add_argument(
        "voice",
        help="Edge-TTS voice, e.g. en-US-AndrewNeural",
    )

    parser.add_argument(
        "audio_out",
        help="Output MP3 path",
    )

    parser.add_argument(
        "timing_out",
        help="Output timing JSON path",
    )

    parser.add_argument(
        "--rate",
        default="+0%",
        help='Speech rate, e.g. "-4%" or "+5%"',
    )

    parser.add_argument(
        "--pitch",
        default="+0Hz",
        help='Voice pitch, e.g. "-2Hz" or "+2Hz"',
    )

    args = parser.parse_args()

    script_text = Path(
        args.text_file
    ).read_text(encoding="utf-8")

    synthesize_speech(
        text=script_text,
        voice=args.voice,
        audio_out=args.audio_out,
        timing_out=args.timing_out,
        rate=args.rate,
        pitch=args.pitch,
    )

    print(
        f"Wrote {args.audio_out} and {args.timing_out}"
    )
