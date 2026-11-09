"""
main.py
Orchestrates one full run of the pipeline: pick topic -> generate script ->
TTS -> fetch stock footage -> assemble video -> upload to YouTube.

Usage:
    python src/main.py mind_bites --short
    python src/main.py mind_bites          # long-form

This is what the GitHub Actions workflow calls, once per video, on a schedule.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import os
import random
import re
import sys
import tempfile
import traceback
from pathlib import Path

import yaml

from src.generate_script import (
    generate_script,
    generate_kids_script,
    save_todays_longform_topic,
    get_recap_topic_for_short,
)
from src.quality_gate import validate_and_repair
from src.tts import synthesize_speech
from src.fetch_stock import fetch_clips_for_topic, fetch_clips_for_shots
from src.assemble_video import assemble_video
from src.generate_thumbnail import generate_thumbnail
from src.generate_kids_video import generate_kids_video
from src.generate_mascot_assets import generate_mascot_assets
from src.upload_youtube import upload_video

ROOT = Path(__file__).resolve().parent.parent


def load_channel_config(channel_id: str) -> dict:
    with open(ROOT / "channels" / f"{channel_id}.yaml") as f:
        return yaml.safe_load(f)


def make_title(title: str, is_short: bool) -> str:
    title = title.strip()

    if title and is_short and not title.endswith("!") and not title.endswith("?"):
        title = title.rstrip(".")

    return title[:95] + (" #Shorts" if is_short else "")


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "why", "what", "when", "where", "which", "who", "how",
    "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "your", "you", "it", "its", "that", "this", "than",
    "actually", "really", "just",
}


def _extract_hashtags_from_title(title: str, max_tags: int = 4) -> list:
    """Pull distinctive words from the title for topic-specific hashtags."""

    words = re.findall(r"[A-Za-z]+", title)

    seen = set()
    tags = []

    for w in words:
        lw = w.lower()

        if lw in _STOPWORDS or len(w) < 4 or lw in seen:
            continue

        seen.add(lw)
        tags.append(w.capitalize())

        if len(tags) >= max_tags:
            break

    return tags


def build_description(
    title: str,
    script_text: str,
    topic: str,
    channel_tags: list,
    is_short: bool,
    source_credits: list | None = None,
) -> str:
    """Build the YouTube description."""

    hook = f"{title.strip()} 👀"

    extracted = _extract_hashtags_from_title(title)

    all_tag_words = list(
        dict.fromkeys(channel_tags + extracted)
    )

    hashtags = " ".join(
        "#" + t.replace(" ", "")
        for t in all_tag_words[:12]
    )

    if is_short:
        hashtags += " #Shorts #ShortVideo"

    cta = "Subscribe for more amazing animal facts every day! 🐾"

    description = (
        f"{hook}\n\n"
        f"{script_text.strip()}\n\n"
        f"{cta}\n\n"
        f"{hashtags}"
    )

    if source_credits:
        attribution_lines = []
        for item in source_credits:
            title_text = str(item.get("title") or "Wikimedia Commons video").strip()
            author = str(item.get("author") or "Author not specified").strip()
            license_name = str(item.get("license") or "License details on source page").strip()
            source_page = str(item.get("source_page") or "").strip()
            license_url = str(item.get("license_url") or "").strip()
            line = f'- "{title_text}" by {author} | {license_name}'
            if source_page:
                line += f" | Source: {source_page}"
            if license_url:
                line += f" | License: {license_url}"
            attribution_lines.append(line)

        description += "\n\nVIDEO FOOTAGE CREDITS (WIKIMEDIA COMMONS)\n" + "\n".join(
            attribution_lines
        )

    return description


MAX_FULL_PIPELINE_ATTEMPTS = 5


class QualityGateRejectedError(RuntimeError):
    """Raised only when the final script Quality Gate rejects a full attempt."""


def _run_once(
    channel_id: str,
    is_short: bool,
    privacy_status: str = "public",
    dry_run: bool = False,
) -> None:

    config = load_channel_config(channel_id)

    # Cosmic Curious routes text tasks through DeepSeek; image-based Visual QA uses Qwen.
    os.environ["AI_PROVIDER"] = "freeinference"


    # Long-form is globally disabled. Reject manual/default invocations
    # before authentication, topic selection, or any FreeInference API calls.
    if not is_short:
        raise RuntimeError(
            "Long-form is disabled. Run the pipeline with --short."
        )

    is_kids_channel = (
        config.get("visual_style") == "puppet_animation"
    )

    print(
        f"=== Running {config['display_name']} | "
        f"{'SHORT' if is_short else 'LONG-FORM'} ==="
    )

    # ---------------------------------------------------------
    # AUTH
    # ---------------------------------------------------------

    if not dry_run:

        from src.upload_youtube import verify_token_valid

        token_path = ROOT / "tokens" / f"{channel_id}_token.pickle"
        client_secret_path = ROOT / "client_secret.json"

        verify_token_valid(
            str(token_path),
            str(client_secret_path)
            if client_secret_path.exists()
            else None,
        )

        print("Pre-flight auth check passed.")

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        # -----------------------------------------------------
        # 1. SCRIPT
        # -----------------------------------------------------

        if is_kids_channel:

            result = generate_kids_script(
                channel_id,
                is_short=is_short,
            )

        else:

            forced_topic = (
                get_recap_topic_for_short(channel_id)
                if is_short
                else None
            )

            result = generate_script(
                channel_id,
                is_short=is_short,
                forced_topic=forced_topic,
            )

            if not is_short:

                save_todays_longform_topic(
                    channel_id,
                    result["topic"],
                )

        topic = result["topic"]
        script_text = result["script"]

        language = result["language"]
        voice = result["voice"]

        print(f"Topic: {topic}")

        if is_kids_channel:
            print(
                f"Content type: {result['content_type']} | "
                f"Mascot: {result['mascot_name']}"
            )

        print(
            f"Language: {language} | "
            f"Voice: {voice}"
        )

        # -----------------------------------------------------
        # EDITORIAL PASS
        # -----------------------------------------------------

        editorial_title = result["title"]

        extra_tags = []
        extra_hashtags = []
        thumbnail_override_text = None

        if not is_kids_channel:

            from src.editorial_pass import run_editorial_pass

            edit_result = run_editorial_pass(
                topic,
                script_text,
                editorial_title,
                config,
            )

            script_text = edit_result["script"]
            editorial_title = edit_result["title"]
            extra_tags = edit_result["extra_tags"]
            extra_hashtags = edit_result["extra_hashtags"]
            thumbnail_override_text = edit_result["thumbnail_text"]

        # -----------------------------------------------------
        # FINAL QUALITY GATE
        # -----------------------------------------------------
        # Run after the editorial agents because they can rewrite the
        # opening sentence and body. Nothing may reach TTS unless the
        # exact final narration has passed the scientific review.

        if not is_kids_channel:

            target_words = int(
                config["short_length_seconds"] * 2.2
            )

            script_text, quality_review = validate_and_repair(
                topic=topic,
                script=script_text,
                config=config,
                target_words=target_words,
            )

            result["script"] = script_text

            print(
                f"Final Quality score: "
                f"{quality_review.get('score', 0):.1f}/10"
            )

            if not quality_review.get("approved"):
                raise QualityGateRejectedError(
                    "Final Quality Gate rejected the edited script. "
                    "TTS and publishing are blocked for this attempt."
                )

        print(
            f"Final script ({len(script_text.split())} words):\n"
            f"{script_text}\n"
        )

        # -----------------------------------------------------
        # 2. TTS
        # -----------------------------------------------------

        audio_path = tmp / "audio.mp3"
        timing_path = tmp / "timing.json"

        rate = f"{random.randint(-6, -2)}%"

        synthesize_speech(
            script_text,
            voice,
            str(audio_path),
            str(timing_path),
            rate=rate,
        )

        print(
            f"Audio generated: {audio_path}"
        )

        # -----------------------------------------------------
        # 3+4. VISUALS + ASSEMBLY
        # -----------------------------------------------------

        output_path = tmp / "final.mp4"

        if is_kids_channel:

            mascot_dir = tmp / "mascot_assets"

            frame_paths = generate_mascot_assets(
                result["mascot"],
                str(mascot_dir),
            )

            print(
                f"Mascot assets ready: {frame_paths}"
            )

            generate_kids_video(
                str(audio_path),
                str(timing_path),
                frame_paths,
                str(output_path),
                portrait=is_short,
            )

        else:

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Short = 15 visual clips
            # Long  = 30 visual clips
            #
            # assemble_video.py distributes the audio duration
            # across these clips, resulting in approximately
            # 3-second visual changes for a 45-second Short
            # and approximately 10 seconds for a 300-second
            # long-form video.
            # -------------------------------------------------

            clip_count = 15 if is_short else 30

            clips_dir = tmp / "clips"

            orientation = "portrait" if is_short else "landscape"

            if channel_id == "cosmic_curious":
                from src.agents.visual_agent import plan as plan_visuals

                print("Cosmic Curious: building narration-synced animal shot plan...")

                clip_paths = None

                try:
                    visual_plan = plan_visuals(
                        topic,
                        script_text,
                        shot_count=clip_count,
                        title=editorial_title,
                    )
                    shots = (
                        visual_plan.get("shots", [])
                        if isinstance(visual_plan, dict)
                        else []
                    )

                    if len(shots) != clip_count:
                        raise RuntimeError(
                            "Visual Director did not return exactly "
                            f"{clip_count} shots (received {len(shots)}). "
                            "Upload blocked."
                        )

                    clip_paths = fetch_clips_for_shots(
                        shots,
                        str(clips_dir),
                        orientation=orientation,
                    )

                except Exception as exc:
                    raise RuntimeError(
                        "Cosmic Curious visual sourcing failed; refusing to "
                        "publish a video with unrelated stock footage. "
                        f"Details: {str(exc)[:300]}"
                    ) from exc

                if not clip_paths:
                    raise RuntimeError(
                        "Cosmic Curious visual sourcing returned no verified "
                        "animal clips; upload blocked."
                    )
            else:
                clip_paths = fetch_clips_for_topic(
                    topic,
                    str(clips_dir),
                    count=clip_count,
                    orientation=orientation,
                )

            print(
                f"Downloaded {len(clip_paths)} clips"
            )

            assemble_video(
                clip_paths,
                str(audio_path),
                str(timing_path),
                str(output_path),
                portrait=is_short,
            )

        print(
            f"Assembled video: {output_path} "
            f"({output_path.stat().st_size / 1e6:.1f} MB)"
        )

        # -----------------------------------------------------
        # DRY RUN
        # -----------------------------------------------------

        if dry_run:

            dest = (
                ROOT
                / "output"
                / f"{channel_id}_"
                f"{'short' if is_short else 'long'}.mp4"
            )

            dest.parent.mkdir(
                exist_ok=True
            )

            dest.write_bytes(
                output_path.read_bytes()
            )

            print(
                f"[DRY RUN] Saved to {dest} "
                "instead of uploading"
            )

            return

        # -----------------------------------------------------
        # 5. THUMBNAIL
        # -----------------------------------------------------

        thumb_path = tmp / "thumbnail.jpg"

        thumbnail_result = generate_thumbnail(
            str(output_path),
            editorial_title,
            str(thumb_path),
            language=language,
            portrait=is_short,
            override_text=thumbnail_override_text,
        )

        # -----------------------------------------------------
        # 5b. QA
        # -----------------------------------------------------

        from src.qa_check import validate_video

        ok, fatal_issues, qa_warnings = validate_video(
            str(output_path),
            str(audio_path),
            str(thumb_path)
            if thumbnail_result
            else None,
        )

        for warning in qa_warnings:
            print(
                f"QA WARNING: {warning}"
            )

        if not ok:

            issues_text = "; ".join(
                fatal_issues
            )

            raise RuntimeError(
                f"QA check failed, blocking upload: "
                f"{issues_text}"
            )

        print("QA check passed.")

        # -----------------------------------------------------
        # 6. UPLOAD
        # -----------------------------------------------------

        title = make_title(
            editorial_title,
            is_short,
        )

        source_credits = []
        if channel_id == "cosmic_curious":
            credits_path = clips_dir / "source_credits.json"
            if credits_path.exists():
                try:
                    loaded_credits = json.loads(credits_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_credits, list):
                        source_credits = [
                            item for item in loaded_credits if isinstance(item, dict)
                        ]
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Could not read Wikimedia Commons attribution metadata: {exc}"
                    ) from exc

        description = build_description(
            editorial_title,
            script_text,
            topic,
            config["tags"] + extra_tags,
            is_short,
            source_credits=source_credits,
        )

        video_tags = list(
            dict.fromkeys(
                config["tags"]
                + extra_tags
                + extra_hashtags
                + _extract_hashtags_from_title(
                    editorial_title,
                    max_tags=6,
                )
            )
        )

        token_path = (
            ROOT
            / "tokens"
            / f"{channel_id}_token.pickle"
        )

        video_id = upload_video(
            str(output_path),
            title,
            description,
            video_tags,
            config["category_id"],
            str(token_path),
            privacy_status=privacy_status,
            is_short=is_short,
            thumbnail_path=thumbnail_result,
            made_for_kids=config.get(
                "made_for_kids",
                False,
            ),
        )

        print(
            f"Uploaded: "
            f"https://youtube.com/watch?v={video_id}"
        )

        # -----------------------------------------------------
        # VIDEO REGISTRY
        # -----------------------------------------------------

        try:

            from src.agents.video_registry import record_video

            record_video(
                channel_id,
                {
                    "video_id": video_id,
                    "topic": topic,
                    "title": title,
                    "is_short": is_short,
                    "language": language,
                },
            )

        except Exception as e:

            print(
                "WARNING: video_registry logging failed "
                f"({e}) -- upload itself succeeded."
            )



def run(
    channel_id: str,
    is_short: bool,
    privacy_status: str = "public",
    dry_run: bool = False,
) -> None:
    """
    Restart the complete video-generation attempt when the final
    Quality Gate rejects a script. Each attempt gets a fresh temporary
    directory and regenerates the topic, script, editorial pass and visuals.
    """
    for attempt in range(1, MAX_FULL_PIPELINE_ATTEMPTS + 1):
        try:
            return _run_once(
                channel_id,
                is_short,
                privacy_status,
                dry_run,
            )
        except QualityGateRejectedError as exc:
            if attempt >= MAX_FULL_PIPELINE_ATTEMPTS:
                raise RuntimeError(
                    f"Quality Gate rejected all {MAX_FULL_PIPELINE_ATTEMPTS} "
                    f"complete pipeline attempts. Publishing is blocked. "
                    f"Last rejection: {exc}"
                ) from exc

            print(
                f"QUALITY GATE REJECTED attempt {attempt}/"
                f"{MAX_FULL_PIPELINE_ATTEMPTS}. "
                f"Restarting the entire pipeline from scratch "
                f"({attempt + 1}/{MAX_FULL_PIPELINE_ATTEMPTS})."
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "channel_id"
    )

    parser.add_argument(
        "--short",
        action="store_true",
    )

    parser.add_argument(
        "--privacy",
        default="public",
        choices=[
            "public",
            "unlisted",
            "private",
        ],
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the video but don't upload — "
            "saves locally instead"
        ),
    )

    args = parser.parse_args()

    try:

        run(
            args.channel_id,
            args.short,
            args.privacy,
            args.dry_run,
        )

    except Exception:

        print(
            "PIPELINE FAILED:",
            file=sys.stderr,
        )

        traceback.print_exc()

        sys.exit(1)