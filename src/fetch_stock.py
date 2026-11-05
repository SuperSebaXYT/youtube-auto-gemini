"""
fetch_stock.py

Smart Pexels stock-video retrieval for the AutoTube pipeline.

V4:
- Searches multiple topic-specific queries.
- Understands special animal terms such as "sea cucumber".
- Collects a larger candidate pool.
- Prefers relevant/orientation-matching videos.
- Avoids duplicate Pexels IDs.
- Avoids downloading the same URL twice.
- Supports larger clip counts for frequent visual changes.
- Keeps downloaded files at a reasonable resolution.
- Falls back gracefully when the main search returns too few results.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests


PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to",
    "why", "what", "how", "your", "you", "we",
    "and", "is", "was", "it", "that", "for",
    "when", "are", "were", "be", "been",
    "being", "do", "does", "did", "can",
    "could", "will", "would", "should",
    "shall", "may", "might", "must",
    "has", "have", "had", "this", "these",
    "those", "from", "near", "where", "which",
    "who", "whom", "with", "by", "at", "as",
    "if", "than", "then", "so", "not", "no",
    "actually", "really", "just", "one",
    "single", "every", "any", "some",
    "possible", "possibly", "or", "but",
    "there", "here", "reason", "its",
    "it's", "happen", "happens",
    "happening",
}


def _topic_words(topic: str) -> list:
    """Extract meaningful words from the topic."""

    words = re.findall(
        r"[a-zA-Z]+",
        topic.lower(),
    )

    return [
        word
        for word in words
        if word not in STOPWORDS
        and len(word) >= 3
    ]


def _is_sea_cucumber_topic(topic: str) -> bool:
    """
    Detect the animal 'sea cucumber'.

    This is important because a plain Pexels search for
    'sea cucumber' can return footage of ordinary cucumbers.
    """

    normalized = re.sub(
        r"[^a-zA-Z]+",
        " ",
        topic.lower(),
    ).strip()

    return bool(
        re.search(
            r"\bsea\s+cucumber(s)?\b",
            normalized,
        )
    )


def _is_specific_animal_topic(topic: str) -> bool:
    """
    Detect animal topics that should stay focused on wildlife
    rather than falling back to unrelated generic footage.
    """

    words = set(
        _topic_words(topic)
    )

    if _is_sea_cucumber_topic(topic):
        return True

    animal_words = {
        "animal",
        "animals",
        "wildlife",
        "mammal",
        "mammals",
        "bird",
        "birds",
        "reptile",
        "reptiles",
        "snake",
        "snakes",
        "lion",
        "lions",
        "tiger",
        "tigers",
        "bear",
        "bears",
        "wolf",
        "wolves",
        "fox",
        "monkey",
        "monkeys",
        "elephant",
        "elephants",
        "giraffe",
        "giraffes",
        "zebra",
        "zebras",
        "shark",
        "sharks",
        "whale",
        "whales",
        "dolphin",
        "dolphins",
        "penguin",
        "penguins",
        "frog",
        "frogs",
        "fish",
        "octopus",
        "spider",
        "spiders",
        "insect",
        "insects",
        "wild",
        "nature",
        "ocean",
        "jungle",
        "savanna",
        "safari",
    }

    return bool(
        words.intersection(animal_words)
    )


def _build_search_queries(topic: str) -> list:
    """
    Build several searches ordered from specific to broad.

    Special animal topics receive semantic queries designed
    to prevent ambiguous searches from returning unrelated
    objects.
    """

    words = _topic_words(topic)

    queries = []

    # ---------------------------------------------------------
    # SPECIAL CASE: SEA CUCUMBER
    # ---------------------------------------------------------

    if _is_sea_cucumber_topic(topic):

        queries.extend([
            "sea cucumber animal",
            "sea cucumber underwater",
            "sea cucumber ocean",
            "sea cucumber marine animal",
            "holothurian",
            "holothurian underwater",
            "holothurian ocean",
            "sea cucumber wildlife",
        ])

        # Do NOT start with the ambiguous plain query:
        # "sea cucumber"
        #
        # The word "cucumber" can cause stock sites to return
        # footage of ordinary vegetables.

    else:

        # -----------------------------------------------------
        # NORMAL TOPIC QUERIES
        # -----------------------------------------------------

        if words:

            queries.append(
                " ".join(words[:4])
            )

        if len(words) >= 2:

            queries.append(
                " ".join(words[:3])
            )

    # ---------------------------------------------------------
    # ANIMAL / WILDLIFE
    # ---------------------------------------------------------

    animal_words = {
        "animal",
        "animals",
        "wildlife",
        "mammal",
        "mammals",
        "bird",
        "birds",
        "reptile",
        "reptiles",
        "snake",
        "snakes",
        "lion",
        "lions",
        "tiger",
        "tigers",
        "bear",
        "bears",
        "wolf",
        "wolves",
        "fox",
        "monkey",
        "monkeys",
        "elephant",
        "elephants",
        "giraffe",
        "giraffes",
        "zebra",
        "zebras",
        "shark",
        "sharks",
        "whale",
        "whales",
        "dolphin",
        "dolphins",
        "penguin",
        "penguins",
        "frog",
        "frogs",
        "fish",
        "octopus",
        "spider",
        "spiders",
        "insect",
        "insects",
        "wild",
        "nature",
        "ocean",
        "jungle",
        "savanna",
        "safari",
    }

    if (
        any(
            word in animal_words
            for word in words
        )
        or _is_sea_cucumber_topic(topic)
    ):

        queries.extend([
            "wildlife animals",
            "wild animals",
            "animal behavior",
            "animals nature",
            "wildlife nature",
            "animals close up",
            "animals documentary",
            "wildlife documentary",
        ])

    # ---------------------------------------------------------
    # OLD SPACE / SCIENCE SUPPORT
    # ---------------------------------------------------------

    if "black" in words and "hole" in words:

        queries.extend([
            "black hole space",
            "black hole astronomy",
        ])

    if "star" in words:

        queries.extend([
            "star space",
            "stars astronomy",
        ])

    if "planet" in words:

        queries.extend([
            "planet space",
            "planet astronomy",
        ])

    if "neutron" in words:

        queries.extend([
            "neutron star",
            "neutron star space",
        ])

    if "galaxy" in words:

        queries.extend([
            "galaxy space",
            "galaxy astronomy",
        ])

    if "gravity" in words:

        queries.extend([
            "gravity space",
            "gravity physics",
        ])

    if "moon" in words:

        queries.extend([
            "moon space",
            "moon astronomy",
        ])

    if "sun" in words:

        queries.extend([
            "sun space",
            "sun astronomy",
        ])

    # ---------------------------------------------------------
    # GENERAL FALLBACK
    # ---------------------------------------------------------

    queries.extend([
        "nature wildlife",
        "animals nature",
        "documentary nature",
        "cinematic nature",
        "nature close up",
    ])

    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(queries)
    )


def _search_pexels(
    query: str,
    headers: dict,
    orientation: str,
    per_page: int = 15,
    page: int = 1,
) -> list:
    """Search Pexels and return videos."""

    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": orientation,
        "size": "medium",
    }

    response = requests.get(
        PEXELS_SEARCH_URL,
        headers=headers,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json().get(
        "videos",
        [],
    )


def _choose_video_file(video: dict):
    """
    Select a sensible resolution.

    Prefer roughly 720p-1080p to keep the automated
    pipeline reasonably fast.
    """

    files = video.get(
        "video_files",
        [],
    )

    if not files:
        return None

    suitable = [
        item
        for item in files
        if 720 <= item.get("width", 0) <= 1920
    ]

    if suitable:

        return sorted(
            suitable,
            key=lambda item: (
                item.get("width", 0),
                item.get("height", 0),
            ),
        )[0]

    return sorted(
        files,
        key=lambda item: item.get("width", 0),
    )[0]


def _score_video(
    video: dict,
    orientation: str,
) -> float:
    """
    Lightweight visual suitability score.
    """

    width = video.get("width") or 0
    height = video.get("height") or 0
    duration = video.get("duration") or 0

    if width <= 0 or height <= 0:
        return 0

    score = 0

    ratio = width / height

    if orientation == "portrait":

        if ratio < 1:
            score += 5

        elif ratio < 1.2:
            score += 3

    else:

        if ratio > 1.4:
            score += 5

        elif ratio > 1.1:
            score += 3

    if 5 <= duration <= 30:
        score += 2

    elif duration > 30:
        score += 1

    if width >= 720:
        score += 2

    return score


def fetch_clips_for_topic(
    topic: str,
    out_dir: str,
    count: int = 5,
    orientation: str = "landscape",
) -> list:
    """
    Download stock clips relevant to the topic.

    A larger candidate pool is collected before selecting
    the requested number of unique clips.
    """

    api_key = os.environ.get(
        "PEXELS_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "PEXELS_API_KEY not set. "
            "Get a free key at https://www.pexels.com/api/"
        )

    headers = {
        "Authorization": api_key,
    }

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    queries = _build_search_queries(
        topic
    )

    videos_by_id = {}

    specific_animal = (
        _is_specific_animal_topic(topic)
    )

    # ---------------------------------------------------------
    # SEARCH MULTIPLE QUERIES
    # ---------------------------------------------------------

    target_candidates = max(
        count * 3,
        30,
    )

    for query in queries:

        for page in (1, 2):

            try:

                videos = _search_pexels(
                    query,
                    headers,
                    orientation,
                    per_page=15,
                    page=page,
                )

            except Exception as e:

                print(
                    f"WARNING: Pexels search failed "
                    f"for '{query}' page {page} ({e})"
                )

                continue

            for video in videos:

                video_id = video.get(
                    "id"
                )

                if video_id:

                    videos_by_id[
                        video_id
                    ] = video

            if len(videos_by_id) >= target_candidates:
                break

        if len(videos_by_id) >= target_candidates:
            break

    # ---------------------------------------------------------
    # FALLBACK SEARCH
    # ---------------------------------------------------------

    if len(videos_by_id) < count:

        if _is_sea_cucumber_topic(topic):

            # Never use generic cucumber searches.
            #
            # If Pexels has too few sea-cucumber clips,
            # prefer underwater/ocean footage instead of
            # showing an ordinary vegetable.

            fallback_queries = [
                "underwater marine life",
                "ocean animals",
                "underwater wildlife",
                "marine animals",
                "ocean floor",
                "coral reef animals",
            ]

        elif specific_animal:

            fallback_queries = [
                "wildlife",
                "animals",
                "wild animals",
                "nature animals",
                "animal close up",
                "nature documentary",
            ]

        else:

            fallback_queries = [
                "wildlife",
                "animals",
                "nature",
                "animals close up",
                "wild animals",
                "nature documentary",
            ]

        for query in fallback_queries:

            try:

                videos = _search_pexels(
                    query,
                    headers,
                    orientation,
                    per_page=15,
                    page=1,
                )

                for video in videos:

                    video_id = video.get(
                        "id"
                    )

                    if video_id:

                        videos_by_id[
                            video_id
                        ] = video

                if len(videos_by_id) >= target_candidates:
                    break

            except Exception as e:

                print(
                    f"WARNING: fallback Pexels "
                    f"search failed for "
                    f"'{query}' ({e})"
                )

    if not videos_by_id:

        raise RuntimeError(
            f"Pexels returned no usable videos "
            f"for topic: {topic}"
        )

    # ---------------------------------------------------------
    # RANK
    # ---------------------------------------------------------

    ranked = sorted(
        videos_by_id.values(),
        key=lambda video: _score_video(
            video,
            orientation,
        ),
        reverse=True,
    )

    # ---------------------------------------------------------
    # DOWNLOAD
    # ---------------------------------------------------------

    downloaded = []

    used_urls = set()

    for video in ranked:

        if len(downloaded) >= count:
            break

        target = _choose_video_file(
            video
        )

        if not target:
            continue

        url = target.get(
            "link"
        )

        if not url:
            continue

        if url in used_urls:
            continue

        used_urls.add(url)

        clip_index = len(downloaded)

        clip_path = (
            out_dir
            / f"clip_{clip_index:02d}.mp4"
        )

        try:

            with requests.get(
                url,
                stream=True,
                timeout=60,
            ) as response:

                response.raise_for_status()

                with open(
                    clip_path,
                    "wb",
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 64
                    ):

                        if chunk:
                            f.write(chunk)

            if (
                clip_path.exists()
                and clip_path.stat().st_size > 10000
            ):

                downloaded.append(
                    str(clip_path)
                )

        except Exception as e:

            print(
                "WARNING: failed downloading "
                f"Pexels clip ({e})"
            )

            if clip_path.exists():

                try:
                    clip_path.unlink()

                except OSError:
                    pass

    if not downloaded:

        raise RuntimeError(
            "Could not download usable Pexels "
            f"footage for topic: {topic}"
        )

    print(
        f"Pexels: topic={topic!r}"
    )

    print(
        "Pexels queries tried: "
        + " | ".join(
            queries[:10]
        )
    )

    print(
        f"Pexels: collected "
        f"{len(videos_by_id)} candidates"
    )

    print(
        f"Pexels: selected "
        f"{len(downloaded)} clips"
    )

    return downloaded


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "topic"
    )

    parser.add_argument(
        "out_dir"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--portrait",
        action="store_true",
    )

    args = parser.parse_args()

    orientation = (
        "portrait"
        if args.portrait
        else "landscape"
    )

    paths = fetch_clips_for_topic(
        args.topic,
        args.out_dir,
        args.count,
        orientation,
    )

    print(
        "\n".join(paths)
    )


def _download_pexels_video(video: dict, clip_path: Path) -> bool:
    """Download one Pexels candidate to clip_path."""
    target = _choose_video_file(video)
    if not target:
        return False

    url = target.get("link")
    if not url:
        return False

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(clip_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        output.write(chunk)

        if clip_path.exists() and clip_path.stat().st_size > 10000:
            return True
    except Exception as exc:
        print(f"WARNING: Pexels candidate download failed ({exc})")

    try:
        clip_path.unlink(missing_ok=True)
    except OSError:
        pass
    return False


def _extract_audit_frames(clip_path: Path, frame_dir: Path) -> list[Path]:
    """Extract five evenly spaced frames so QA checks actual clip content."""
    frame_dir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Could not inspect downloaded clip: {probe.stderr[-500:]}")

    duration = float(probe.stdout.strip())
    if duration <= 0:
        raise RuntimeError("Downloaded clip has an invalid duration.")

    frames = []
    for index, fraction in enumerate((0.10, 0.30, 0.50, 0.70, 0.90)):
        timestamp = min(max(0.05, duration * fraction), max(0.05, duration - 0.05))
        frame_path = frame_dir / f"frame_{index}.jpg"

        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{timestamp:.2f}",
                "-i", str(clip_path),
                "-frames:v", "1",
                "-vf", "scale=640:-2",
                "-q:v", "4",
                str(frame_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 1000:
            frames.append(frame_path)

    if not frames:
        raise RuntimeError("Could not extract any frames for visual verification.")

    return frames


def _verify_clip_matches_shot(
    clip_path: Path,
    animal: str,
    action: str,
    sentence: str,
    visual: str,
) -> tuple[bool, str]:
    """Use the configured vision model to verify downloaded video frames."""
    from src.gemini_client import generate_vision, is_available

    if not is_available():
        provider = os.environ.get("AI_PROVIDER", "gemini").upper()
        raise RuntimeError(f"{provider} API key is required for Cosmic Curious visual verification.")

    with tempfile.TemporaryDirectory(prefix="cosmic_visual_qa_") as temp_dir:
        frames = _extract_audit_frames(clip_path, Path(temp_dir))
        prompt = f"""
You are a strict visual evidence inspector for a wildlife video.
Inspect the attached frames from ONE real video clip.

NARRATION SEGMENT:
{sentence}

REQUIRED ANIMAL:
{animal}

REQUIRED VISIBLE ACTION:
{action}

INTENDED SHOT:
{visual}

Approve ONLY if:
1. The required animal is clearly visible and is the main subject.
2. The frames visibly show the required action/behavior, not merely an
   animal that could theoretically perform it.
3. There is no unrelated object or food being used as a substitute.
4. The visual evidence is clear enough to identify the animal and action.

Reject pasta, food, cooking, liquid in a bowl, unrelated animals, generic
nature scenery, empty habitats, and clips where the correct behavior is
not visible. Do not infer the action from the narration. Judge only what
is actually visible in the frames.

Return ONLY valid JSON:
{{
  "animal_visible": true,
  "animal_identification": "what is visibly shown",
  "action_visible": true,
  "confidence": 0.95,
  "reason": "brief evidence-based explanation"
}}
"""
        response = generate_vision(
            prompt,
            [(frame_path.read_bytes(), "image/jpeg") for frame_path in frames],
        )
        raw = (getattr(response, "text", None) or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Vision model returned invalid JSON: {raw[:300]}")

        result = json.loads(match.group(0))
        animal_ok = result.get("animal_visible") is True
        action_ok = result.get("action_visible") is True

        try:
            confidence = float(result.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(result.get("reason", "No reason returned"))
        approved = animal_ok and action_ok and confidence >= 0.80

        return approved, (
            f"animal_visible={animal_ok}, action_visible={action_ok}, "
            f"confidence={confidence:.2f}: {reason}"
        )


def fetch_clips_for_shots(
    shots: list,
    out_dir: str,
    orientation: str = "portrait",
    max_candidates_per_shot: int = 4,
) -> list:
    """
    Fetch one clip per narration shot, using only that shot's exact animal
    and behavior query. Every downloaded clip is frame-checked by the configured vision model.
    There is deliberately NO generic wildlife/nature fallback. If a matching
    clip cannot be found and verified, the pipeline stops before upload.
    """
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY not set.")

    if not shots:
        raise RuntimeError("Visual Director returned an empty shot list.")

    headers = {"Authorization": api_key}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    used_video_ids = set()

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Visual shot {index + 1} is not a JSON object.")

        animal = str(shot.get("animal", "")).strip()
        action = str(shot.get("action", "")).strip()
        sentence = str(shot.get("sentence", "")).strip()
        visual = str(shot.get("visual", "")).strip()
        query = str(shot.get("search_query", "")).strip()

        if not all((animal, action, sentence, visual, query)):
            raise RuntimeError(
                f"Visual shot {index + 1} is missing animal/action/query details."
            )

        # Prevent the director from silently reverting to broad stock searches.
        query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
        animal_words = set(re.findall(r"[a-zA-Z]+", animal.lower()))
        if not animal_words.intersection(query_words):
            raise RuntimeError(
                f"Visual shot {index + 1} query does not name its required animal: {query!r}"
            )

        print(
            f"Visual shot {index + 1}/{len(shots)}: "
            f"animal={animal!r} action={action!r} query={query!r}"
        )

        candidates = _search_pexels(
            query,
            headers,
            orientation,
            per_page=15,
            page=1,
        )
        ranked = sorted(
            candidates,
            key=lambda video: _score_video(video, orientation),
            reverse=True,
        )

        accepted_path = None
        checked = 0

        for video in ranked:
            video_id = video.get("id")
            if not video_id or video_id in used_video_ids:
                continue

            checked += 1
            if checked > max_candidates_per_shot:
                break

            candidate_path = out_path / f"candidate_{index:02d}_{checked:02d}.mp4"
            if not _download_pexels_video(video, candidate_path):
                continue

            try:
                approved, audit_reason = _verify_clip_matches_shot(
                    candidate_path,
                    animal=animal,
                    action=action,
                    sentence=sentence,
                    visual=visual,
                )
            except Exception as exc:
                candidate_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Visual verification failed for shot {index + 1}; "
                    f"upload blocked. Details: {exc}"
                ) from exc

            if approved:
                accepted_path = out_path / f"clip_{index:02d}.mp4"
                candidate_path.replace(accepted_path)
                used_video_ids.add(video_id)
                print(f"Visual QA PASSED for shot {index + 1}: {audit_reason}")
                break

            print(f"Visual QA REJECTED candidate for shot {index + 1}: {audit_reason}")
            candidate_path.unlink(missing_ok=True)

        if accepted_path is None:
            raise RuntimeError(
                f"No verified Pexels clip for shot {index + 1}: "
                f"{animal} — {action}. Pipeline stopped; no unrelated fallback."
            )

        downloaded.append(str(accepted_path))

    print(
        f"Cosmic Curious visual QA: verified {len(downloaded)}/{len(shots)} "
        "animal-and-action-matched clips."
    )
    return downloaded
