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
from concurrent.futures import ThreadPoolExecutor
import time
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


def _query_mentions_target_species(search_query: str, animal: str) -> bool:
    """Return True only when a Pexels query explicitly names the target species."""
    normalized_query = re.sub(r"[^a-z0-9]+", " ", search_query.lower()).strip()
    normalized_animal = re.sub(r"[^a-z0-9]+", " ", animal.lower()).strip()

    accepted_names = {
        "secretary bird": [
            "secretary bird",
            "secretarybird",
            "sagittarius serpentarius",
        ],
        "african fish eagle": [
            "african fish eagle",
            "african sea eagle",
            "fish eagle africa",
            "haliaeetus vocifer",
        ],
    }.get(normalized_animal, [normalized_animal])

    return any(
        re.search(
            r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])",
            normalized_query,
        )
        for name in accepted_names
        if name
    )


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

    # Do not rank by clip length or subjective technical polish.
    # Keep only portrait suitability and enough source resolution for Shorts.
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
    # Three evenly spaced frames are enough for a first-pass audit.
    # This cuts image upload and vision inference cost by 40%.
    for index, fraction in enumerate((0.15, 0.50, 0.85)):
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


def _verify_clip_batch_matches_shot(
    clip_paths: list[Path],
    animal: str,
    strict_species: bool = True,
) -> list[tuple[bool, str]]:
    """Verify up to three clips for the same shot in one vision request."""
    from src.freeinference_client import generate_vision, is_vision_available

    if not clip_paths:
        return []
    if len(clip_paths) > 3:
        raise ValueError("Visual QA batches may contain at most three clips.")
    if not is_vision_available():
        raise RuntimeError("FREEINFERENCE_API_KEY is required for Cosmic Curious visual verification.")

    with tempfile.TemporaryDirectory(prefix="cosmic_visual_batch_") as temp_dir:
        frame_images = []
        for clip_index, clip_path in enumerate(clip_paths, start=1):
            frame_dir = Path(temp_dir) / f"clip_{clip_index:02d}"
            frames = _extract_audit_frames(clip_path, frame_dir)
            frame_images.extend(
                (frame_path.read_bytes(), "image/jpeg") for frame_path in frames
            )

        identity_rule = (
            f'TARGET ANIMAL: {animal}. The exact named animal/species (or an unmistakable '
            "example of that animal type) must be visibly present. Reject a different animal."
            if strict_species else
            "TARGET ANIMAL: any recognizable real wild animal. Reject humans, objects, food, toys, drawings, models, and scenery without a clearly visible animal."
        )
        prompt = f"""
You are checking {len(clip_paths)} separate wildlife video clips for the SAME shot.
{identity_rule}

The images are grouped by clip in order. Each clip has exactly three frames:
clip 1 = images 1-3, clip 2 = images 4-6, clip 3 = images 7-9.
If fewer than three clips are supplied, only evaluate the listed groups.
For EACH clip independently, decide whether the required animal is recognizable,
reasonably clear/prominent, and shown in usable wildlife footage.
Ignore narration, action, pose, habitat, camera angle, and background.
Do not demand rare behavior or impossible species-level details.
Do not let one clip affect another clip's verdict.

Return exactly one line per supplied clip, in this format:
1 MATCH
2 NO_MATCH
3 MATCH
Use only MATCH or NO_MATCH after each number. No JSON or explanation.
"""
        max_attempts = 2
        last_response = ""
        for attempt in range(1, max_attempts + 1):
            try:
                response = generate_vision(prompt, frame_images)
            except Exception as exc:
                error_text = str(exc)
                retryable = any(
                    marker in error_text
                    for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                                   "500", "INTERNAL", "502", "504",
                                   "temporarily unavailable")
                )
                if not retryable or attempt >= max_attempts:
                    raise
                print(
                    f"WARNING: Batch Visual QA temporary API error "
                    f"(attempt {attempt}/{max_attempts}); retrying: {error_text[:250]}"
                )
                time.sleep(2)
                continue

            last_response = (getattr(response, "text", None) or "").strip()
            verdicts = {}
            for line in last_response.splitlines():
                match = re.match(r"^\s*(\d+)\s+(MATCH|NO_MATCH)\s*[.!]*\s*$", line.strip(), re.I)
                if match:
                    verdicts[int(match.group(1))] = match.group(2).upper()

            if set(verdicts) == set(range(1, len(clip_paths) + 1)):
                return [
                    (
                        verdicts[index] == "MATCH",
                        f"animal_type_match={verdicts[index] == 'MATCH'}; vision_verdict={verdicts[index]}"
                    )
                    for index in range(1, len(clip_paths) + 1)
                ]

            print(
                f"WARNING: Batch Visual QA returned incomplete/invalid verdicts "
                f"(attempt {attempt}/{max_attempts}): {last_response[:220]!r}"
            )

        raise RuntimeError(
            f"Batch Visual QA did not return one valid verdict per clip: {last_response[:250]!r}"
        )


def _verify_clip_matches_shot(
    clip_path: Path,
    animal: str,
    action: str,
    sentence: str,
    visual: str,
    require_action: bool = False,
    strict_species: bool = True,
) -> tuple[bool, str]:
    """Verify only that the intended animal/type is visibly present.

    The narration action, pose, habitat, and background are never approval
    requirements. Vision output is a one-word verdict, not JSON.
    """
    from src.freeinference_client import generate_vision, is_vision_available

    if not is_vision_available():
        raise RuntimeError("FREEINFERENCE_API_KEY is required for Cosmic Curious visual verification.")

    with tempfile.TemporaryDirectory(prefix="cosmic_visual_qa_") as temp_dir:
        frames = _extract_audit_frames(clip_path, Path(temp_dir))
        if strict_species:
            identity_rules = f"""
TARGET ANIMAL: {animal}
The exact named animal/species (or an unmistakable example of that animal type)
must be visibly present. Reject a different animal.
"""
        else:
            identity_rules = """
TARGET ANIMAL: ANY REAL WILD ANIMAL
Any recognizable real wild animal is acceptable. It does not need to match
the narration or the planned shot's animal. Reject humans, objects, food,
toys, drawings, models, and scenery without a clearly visible animal.
"""

        prompt = f"""
You are checking wildlife footage for a video editor.

{identity_rules}

Look at the attached frames and answer only this question:
Is an acceptable animal visibly present AND shown in an appealing, usable
wildlife shot?

The animal must be recognizable and reasonably clear/prominent in the frame.
Prefer engaging, well-composed footage; reject animals that are tiny, heavily
obscured, extremely blurry, or barely visible.
Ignore narration, action, pose, habitat, camera angle, and background.
The animal does NOT need to perform any narrated action.

Do not demand proof of rare behavior or impossible visual species-level
details that cannot be seen in the frames.
If the required animal identity rule above is satisfied, answer MATCH.
Otherwise answer NO_MATCH.

Return exactly one of these two lines, with no JSON and no explanation:
MATCH
NO_MATCH
"""
        frame_images = [
            (frame_path.read_bytes(), "image/jpeg")
            for frame_path in frames
        ]

        # A plain one-token verdict avoids fragile JSON generation/parsing.
        # Retry malformed/ambiguous text responses; never let formatting
        # mistakes crash the full 15-shot pipeline.
        # One retry is enough for transient gateway failures; avoid spending ~41s per clip.
        max_vision_attempts = 2
        last_response = ""
        for attempt in range(1, max_vision_attempts + 1):
            try:
                response = generate_vision(prompt, frame_images)
            except Exception as exc:
                error_text = str(exc)
                retryable = any(
                    marker in error_text
                    for marker in (
                        "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
                        "500", "INTERNAL", "502", "504",
                        "temporarily unavailable",
                    )
                )
                if not retryable or attempt >= max_vision_attempts:
                    raise
                delay = 2
                print(
                    f"WARNING: Vision QA temporary API error "
                    f"(attempt {attempt}/{max_vision_attempts}); "
                    f"retrying in {delay}s: {error_text[:250]}"
                )
                time.sleep(delay)
                continue

            last_response = (getattr(response, "text", None) or "").strip()
            verdict_lines = [
                line.strip().upper().strip(" .! *_")
                for line in last_response.splitlines()
                if line.strip()
            ]
            if verdict_lines and verdict_lines[0] in {"MATCH", "NO_MATCH"}:
                matched = verdict_lines[0] == "MATCH"
                return matched, (
                    f"animal_type_match={matched}; "
                    f"vision_verdict={verdict_lines[0]}"
                )

            print(
                f"WARNING: Vision QA returned an invalid plain-text verdict "
                f"(attempt {attempt}/{max_vision_attempts}): "
                f"{last_response[:180]!r}"
            )

        return False, (
            "Vision QA candidate rejected because the model did not return "
            f"MATCH or NO_MATCH after {max_vision_attempts} attempts: "
            f"{last_response[:180]!r}"
        )




def _score_clip_for_shot(
    clip_path: Path,
    animal: str,
) -> tuple[int, str]:
    """Rank fallback footage only by target species, visibility and clip quality."""
    from src.freeinference_client import generate_vision, is_vision_available

    if not is_vision_available():
        raise RuntimeError("FREEINFERENCE_API_KEY is required for fallback clip ranking.")

    with tempfile.TemporaryDirectory(prefix="cosmic_visual_rank_") as temp_dir:
        frames = _extract_audit_frames(clip_path, Path(temp_dir))
        prompt = f"""
You are ranking a rejected wildlife stock clip for fallback use.
TARGET SPECIES: {animal}

IGNORE ALL NARRATION, SCRIPT, SENTENCE, ACTION, SHOT DESCRIPTION,
SEARCH INTENT, POSE, HABITAT AND STORY CONTEXT. They are irrelevant.
Judge the attached video frames ONLY on these three criteria:
1. Correct animal species: the exact target species must be visibly present.
2. Visibility: the animal is clear, prominent and recognizable.
3. Footage quality: the clip is usable, sharp enough and visually clear.

Score from 0 to 100:
- 70 points: exact target species visibly present.
- 20 points: animal visibility and recognizability.
- 10 points: usable visual quality.
If the exact target species is not visible, award zero species-identity points.
Do not reward or penalize the clip for what the animal is doing or for its
relationship to any narration or shot description.
Return one integer from 0 to 100 and nothing else.
"""
        frame_images = [
            (frame_path.read_bytes(), "image/jpeg")
            for frame_path in frames
        ]
        response = generate_vision(prompt, frame_images)
        raw = (getattr(response, "text", None) or "").strip()
        match = re.search(r"\b(100|[1-9]?\d)\b", raw)
        if not match:
            raise RuntimeError(f"Fallback ranking returned no valid score: {raw[:160]!r}")
        return int(match.group(1)), raw[:160]


def fetch_clips_for_shots(
    shots: list,
    out_dir: str,
    orientation: str = "portrait",
) -> list:
    """
    Fetch one Pexels clip per narration shot.

    Each missing slot gets at most three Pexels search rounds. Rejected clips
    are retained across all rounds and pooled across every missing slot.
    AI ranks that shared pool and assigns the best unique clips to the missing
    slots. Fallback selections bypass Visual QA and are not checked again.
    The returned paths are always ordered by shot slot.
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
    fallback_pool = []
    missing_fallback_slots = []

    # Search a bounded first page for a small set of species-specific queries.
    # The old unbounded pagination multiplied API calls for every shot.
    # Each slot searches indefinitely until Visual QA accepts a clip.
    pexels_per_page = 80
    max_queries_per_round = 8
    max_parallel_searches = 8

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Visual shot {index + 1} is not a JSON object.")

        strict_species = True
        animal = str(shot.get("animal", "")).strip() or "wild animal"
        action = str(shot.get("action", "")).strip()
        sentence = str(shot.get("sentence", "")).strip()
        visual = str(shot.get("visual", "")).strip()
        query = str(shot.get("search_query", "")).strip()

        if strict_species and (not animal or not query):
            raise RuntimeError(
                f"Strict Visual shot {index + 1} is missing its target animal or search query."
            )

        if strict_species:
            query_words = set(re.findall(r"[a-zA-Z]+", query.lower()))
            animal_words = set(re.findall(r"[a-zA-Z]+", animal.lower()))
            if not animal_words.intersection(query_words):
                raise RuntimeError(
                    f"Visual shot {index + 1} query does not name its required animal: {query!r}"
                )

        print(
            f"Visual shot {index + 1}/{len(shots)}: "
            f"mode={'EXACT SPECIES' if strict_species else 'ANY WILDLIFE'}; "
            f"animal={animal!r} action={action!r} query={query!r}"
        )

        # Resume safely: retain any existing non-empty clip for this slot.
        accepted_path = out_path / f"clip_{index:02d}.mp4"
        if accepted_path.exists() and accepted_path.stat().st_size > 10000:
            downloaded.append(str(accepted_path))
            print(
                f"Retained existing clip for shot {index + 1}. "
                f"Remaining slots: {len(shots) - len(downloaded)}"
            )
            continue

        accepted_path.unlink(missing_ok=True)

        accepted_path = None
        checked = 0
        attempted_video_ids = set()
        seen_search_queries = set()
        seen_ai_queries = set()
        retry_round = 0
        rejected_candidates = []
        max_search_rounds = 3
        while accepted_path is None and retry_round < max_search_rounds:
            retry_round += 1
            normalized_animal = re.sub(r"[^a-z]+", " ", animal.lower()).strip()
            species_search_aliases = {
                "secretary bird": [
                    "Secretary bird",
                    "Sagittarius serpentarius",
                    "Sagittarius serpentarius bird",
                    "secretarybird",
                    "secretary bird hunting",
                ],
                "african fish eagle": [
                    "African fish eagle",
                    "Haliaeetus vocifer",
                    "African sea eagle",
                    "African fish eagle fishing",
                    "African fish eagle flying",
                    "fish eagle Africa",
                ],
            }

            if strict_species:
                # The first 10 shots are tied to the narration's exact animal.
                aliases = species_search_aliases.get(
                    normalized_animal,
                    [
                        animal,
                        f"{animal} wildlife",
                        f"{animal} close up",
                        f"{animal} in nature",
                        f"{animal} documentary",
                        f"{animal} habitat",
                        f"{animal} animal",
                        f"{animal} macro",
                    ],
                )
                search_modes = [(query, False)]
                search_modes.extend(
                    (candidate_query, False)
                    for candidate_query in aliases[:2]
                    if candidate_query.strip().lower() != query.lower()
                )
            else:
                # The final 5 shots deliberately use varied wildlife footage.
                # Each retry asks Pexels for fresh broad wildlife results.
                search_modes = [
                    (f"wildlife animal close up {retry_round}", False),
                    (f"wild animals natural habitat {retry_round}", False),
                    (f"cinematic wildlife documentary {retry_round}", False),
                    (f"birds mammals reptiles wildlife {retry_round}", False),
                    (f"underwater wildlife animals {retry_round}", False),
                    (f"safari wildlife animals {retry_round}", False),
                    (f"forest wildlife animals {retry_round}", False),
                    (f"wildlife animal portrait {retry_round}", False),
                ]

            # AI-generated Pexels Query Brain: create fresh, natural stock-search
            # phrases from the animal's known appearance and shot context.
            try:
                from src.freeinference_client import generate_response

                rejected_context = ", ".join(sorted(seen_search_queries)[-40:]) or "none yet"
                if strict_species:
                    query_prompt = f"""
You are an expert Pexels wildlife stock-footage search strategist.
TARGET ANIMAL: {animal}
SHOT DESCRIPTION: {visual or "not provided"}
NARRATION CONTEXT: {sentence or action or "not provided"}
ORIGINAL SEARCH QUERY: {query}
Generate 5 distinct, natural English Pexels VIDEO queries. Use varied
recognizable appearance, camera framing and natural habitat descriptions.
EVERY query must explicitly contain "{animal}" or an accepted scientific/common
name. Never omit or replace the target species. Do not invent traits or include
humans, objects, toys, cartoons or unrelated subjects.
Search cycle: {retry_round}. Make the phrases genuinely different from all previously tried queries.
Previously tried queries: {rejected_context}
Return exactly 5 queries, one per line, no numbering or explanations.
"""
                else:
                    query_prompt = f"""
You are an expert Pexels wildlife stock-footage search strategist.
Find varied, visually appealing footage of ANY real wild animals. The footage
does not need to match the narration or a specific species.
Generate 5 distinct natural English Pexels VIDEO queries spanning different
wildlife categories, habitats and camera framings. Do not include humans,
objects, food, toys, cartoons or unrelated scenery.
Previously tried queries: {rejected_context}
Return exactly 5 queries, one per line, no numbering or explanations.
"""
                response = generate_response(query_prompt, max_output_tokens=220)
                generated_queries = []
                for raw_line in (getattr(response, "text", "") or "").splitlines():
                    candidate_query = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip().strip('"\' ')
                    candidate_query = re.sub(r"\s+", " ", candidate_query)
                    normalized_candidate = candidate_query.lower()
                    if (
                        len(candidate_query) >= 8
                        and len(candidate_query.split()) >= 2
                        and normalized_candidate not in seen_ai_queries
                        and normalized_candidate not in {item[0].lower() for item in search_modes}
                        and (
                            not strict_species
                            or _query_mentions_target_species(candidate_query, animal)
                        )
                    ):
                        seen_ai_queries.add(normalized_candidate)
                        generated_queries.append(candidate_query[:100])
                    if len(generated_queries) >= 5:
                        break

                for candidate_query in generated_queries:
                    search_modes.append((candidate_query, False))

                print(
                    f"Pexels Query Brain: generated {len(generated_queries)} fresh queries "
                    f"for {animal!r} in retry round {retry_round}: {generated_queries}"
                )
            except Exception as exc:
                print(
                    f"WARNING: Pexels Query Brain failed for {animal!r} "
                    f"in retry round {retry_round}; using deterministic fallback queries: {exc}"
                )
                fallback_queries = (
                    [
                        f"{animal} underwater close up" if any(term in normalized_animal for term in ("fish", "shark", "ray", "octopus", "jellyfish")) else f"{animal} close up natural habitat cycle {retry_round}",
                        f"large {animal} in natural environment cycle {retry_round}",
                        f"{animal} side view detailed cycle {retry_round}",
                        f"{animal} moving through its habitat cycle {retry_round}",
                        f"wild {animal} portrait cycle {retry_round}",
                        f"{animal} macro texture detail cycle {retry_round}",
                    ]
                    if strict_species
                    else [
                        f"wildlife animal close up retry {retry_round}",
                        f"wild animals natural habitat retry {retry_round}",
                        f"cinematic wildlife documentary retry {retry_round}",
                        f"birds mammals reptiles wildlife retry {retry_round}",
                        f"underwater wildlife animals retry {retry_round}",
                        f"safari wildlife animals retry {retry_round}",
                    ]
                )
                for candidate_query in fallback_queries:
                    if candidate_query.lower() not in seen_ai_queries:
                        seen_ai_queries.add(candidate_query.lower())
                        search_modes.append((candidate_query, False))

            # Enforce species anchoring on EVERY Pexels request, including the
            # original shot query, aliases, AI-generated phrases, and fallbacks.
            # Descriptive wording is welcome, but a query without the target
            # species name or an accepted scientific/common name is discarded.
            # Never spend another Pexels request on a phrase already tried.
            fresh_search_modes = []
            for search_query, require_action in search_modes:
                if strict_species and not _query_mentions_target_species(search_query, animal):
                    print(
                        f"Skipping unanchored Pexels query for shot {index + 1}: "
                        f"{search_query!r} (required species: {animal!r})"
                    )
                    continue
                normalized_search_query = re.sub(r"\s+", " ", search_query.strip().lower())
                if not normalized_search_query or normalized_search_query in seen_search_queries:
                    continue
                seen_search_queries.add(normalized_search_query)
                fresh_search_modes.append((search_query, require_action))
            search_modes = fresh_search_modes[:max_queries_per_round]

            if not search_modes:
                print(
                    f"Pexels produced no new search phrases for shot {index + 1} "
                    f"in search cycle {retry_round}; waiting before generating new phrases."
                )
                time.sleep(2)
                continue

            candidates_by_id = {}
            search_result_counts = {}

            def search_one_mode(item):
                mode_index, (search_query, require_action) = item
                try:
                    results = _search_pexels(
                        search_query,
                        headers,
                        orientation,
                        per_page=pexels_per_page,
                        page=1,
                    )
                    return mode_index, search_query, require_action, results, None
                except Exception as exc:
                    return mode_index, search_query, require_action, [], exc

            # Pexels network searches are independent. Run them concurrently
            # instead of waiting for each species phrase and every result page.
            with ThreadPoolExecutor(
                max_workers=min(max_parallel_searches, len(search_modes))
            ) as search_executor:
                search_batches = list(
                    search_executor.map(search_one_mode, enumerate(search_modes))
                )

            for mode_index, search_query, require_action, results, search_error in search_batches:
                if search_error is not None:
                    print(
                        f"WARNING: Pexels search failed for shot {index + 1} "
                        f"query={search_query!r}: {search_error}"
                    )
                    search_result_counts[search_query] = 0
                    continue

                search_result_counts[search_query] = len(results)
                for video in results:
                    video_id = video.get("id")
                    if (not video_id or video_id in used_video_ids
                            or video_id in attempted_video_ids):
                        continue

                    existing = candidates_by_id.get(video_id)
                    candidate = {
                        "video": video,
                        "require_action": require_action,
                        "query": search_query,
                        "mode_index": mode_index,
                    }
                    if existing is None or (require_action and not existing["require_action"]):
                        candidates_by_id[video_id] = candidate

            # Balance candidates from every animal-specific query.
            candidates_by_mode = {}
            for video_id, candidate in candidates_by_id.items():
                candidates_by_mode.setdefault(candidate["mode_index"], []).append(
                    (video_id, candidate)
                )

            for mode_candidates in candidates_by_mode.values():
                mode_candidates.sort(
                    key=lambda item: _score_video(item[1]["video"], orientation),
                    reverse=True,
                )

            ranked = []
            mode_indexes = sorted(candidates_by_mode)
            depth = 0
            while True:
                added_this_round = False
                for mode_index in mode_indexes:
                    mode_candidates = candidates_by_mode[mode_index]
                    if depth < len(mode_candidates):
                        ranked.append(mode_candidates[depth])
                        added_this_round = True
                if not added_this_round:
                    break
                depth += 1

            new_candidate_count = len(candidates_by_id)
            print(
                f"Visual shot {index + 1} search progress: "
                f"retry_round={retry_round}, previously_checked={checked}, "
                f"new_candidates={new_candidate_count}, "
                f"total_unique_checked_after_round={checked + new_candidate_count}, "
                f"pexels_results={sum(search_result_counts.values())}, "
                f"{'species' if strict_species else 'wildlife'}_queries={len(search_modes)}. "
                f"Per-query counts: {search_result_counts}"
            )

            # Download and visually verify up to three candidates together.
            from concurrent.futures import FIRST_COMPLETED, wait

            batch_size = 3
            candidate_iter = iter(ranked)
            pending = {}
            batch_candidates = []
            executor = ThreadPoolExecutor(max_workers=batch_size)

            def submit_next_candidate():
                """Submit the next unseen Pexels candidate, if available."""
                nonlocal checked
                for video_id, candidate in candidate_iter:
                    if video_id in attempted_video_ids:
                        continue
                    attempted_video_ids.add(video_id)
                    checked += 1
                    candidate_path = out_path / (
                        f"candidate_{index:02d}_{checked:05d}.mp4"
                    )
                    future = executor.submit(
                        _download_pexels_video,
                        candidate["video"],
                        candidate_path,
                    )
                    pending[future] = (video_id, candidate, candidate_path)
                    return True
                return False

            try:
                for _ in range(batch_size):
                    if not submit_next_candidate():
                        break

                while pending and len(batch_candidates) < batch_size:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        video_id, candidate, candidate_path = pending.pop(future)
                        try:
                            was_downloaded = future.result()
                        except Exception as exc:
                            was_downloaded = False
                            print(
                                f"WARNING: parallel download failed for shot {index + 1}, "
                                f"video {video_id}: {exc}"
                            )

                        if was_downloaded:
                            used_video_ids.add(video_id)
                            batch_candidates.append({
                                "video_id": video_id,
                                "candidate": candidate,
                                "path": candidate_path,
                            })
                        else:
                            candidate_path.unlink(missing_ok=True)
                            if len(batch_candidates) + len(pending) < batch_size:
                                submit_next_candidate()

                    # Keep the download window full until three usable clips
                    # have been collected or this round has no more candidates.
                    while len(batch_candidates) + len(pending) < batch_size:
                        if not submit_next_candidate():
                            break

                if batch_candidates:
                    try:
                        batch_verdicts = _verify_clip_batch_matches_shot(
                            [item["path"] for item in batch_candidates],
                            animal=animal,
                            strict_species=strict_species,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Batch Visual QA failed for shot {index + 1}: {exc}"
                        ) from exc

                    matching_candidates = []
                    for item, (qa_passed, qa_reason) in zip(batch_candidates, batch_verdicts):
                        candidate = item["candidate"]
                        if qa_passed:
                            matching_candidates.append(item)
                            print(
                                f"Visual QA MATCH for shot {index + 1}: "
                                f"round={retry_round}; source_query={candidate['query']!r}; "
                                f"{qa_reason}."
                            )
                        else:
                            rejected_candidates.append({
                                "path": item["path"],
                                "video_id": item["video_id"],
                                "query": candidate["query"],
                                "round": retry_round,
                                "qa_reason": qa_reason,
                            })
                            print(
                                f"Visual QA REJECTED candidate for shot {index + 1}: "
                                f"round={retry_round}; source_query={candidate['query']!r}; "
                                f"{qa_reason}. Keeping it for final comparison."
                            )

                    if matching_candidates:
                        # Candidates arrive in Pexels quality-rank order;
                        # keep the first verified match for this shot.
                        selected = matching_candidates[0]
                        accepted_path = out_path / f"clip_{index:02d}.mp4"
                        selected["path"].replace(accepted_path)
                        for item in matching_candidates[1:]:
                            item["path"].unlink(missing_ok=True)
                        print(
                            f"Downloaded Pexels clip for shot {index + 1}: "
                            f"source_query={selected['candidate']['query']!r}; "
                            f"round={retry_round}; batch Visual QA accepted."
                        )
            finally:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                for _, _, candidate_path in pending.values():
                    candidate_path.unlink(missing_ok=True)

            if accepted_path is None:
                print(
                    f"Clip slot {index + 1} still missing. "
                    f"Progress: {checked} distinct videos checked; "
                    f"{len(attempted_video_ids)} unique video IDs checked so far. "
                    f"Search round {retry_round}/{max_search_rounds} exhausted."
                )
                if new_candidate_count == 0:
                    time.sleep(5)

        if accepted_path is not None:
            for item in rejected_candidates:
                item["path"].unlink(missing_ok=True)

        if accepted_path is None:
            # Do not choose a fallback yet. Collect rejected clips from every
            # missing slot, then rank the shared pool against all missing slots.
            usable_rejections = [
                item for item in rejected_candidates
                if not str(item.get("qa_reason", "")).startswith("Visual QA error:")
            ]
            if usable_rejections:
                for item in usable_rejections:
                    item["source_slot"] = index
                fallback_pool.extend(usable_rejections)
                missing_fallback_slots.append(index)
                print(
                    f"Shot {index + 1}: queued {len(usable_rejections)} rejected "
                    "candidate(s) for global best-fit selection."
                )
            else:
                for item in rejected_candidates:
                    item["path"].unlink(missing_ok=True)
                raise RuntimeError(
                    f"No Visual-QA-checked rejected candidates for shot {index + 1} "
                    f"after {max_search_rounds} rounds."
                )
            continue

        downloaded.append(str(accepted_path))
        print(
            f"Clip sourcing progress: {len(downloaded)}/{len(shots)} downloaded; "
            f"{len(shots) - len(downloaded)} slots still need clips."
        )

    # Global fallback assignment: score every rejected clip against every
    # missing narration slot, then greedily assign the highest-scoring unique
    # clip/slot pairs. This avoids choosing a weak fallback independently
    # when a stronger candidate from another failed slot fits better.
    if missing_fallback_slots:
        ranked_pairs = []
        ranking_errors = []
        for slot_index in missing_fallback_slots:
            target = shots[slot_index]
            target_animal = str(target.get("animal", "")).strip()
            for candidate in fallback_pool:
                if candidate.get("slot_ranked_for") == slot_index:
                    continue
                try:
                    score, reason = _score_clip_for_shot(
                        candidate["path"],
                        target_animal,
                    )
                except Exception as exc:
                    ranking_errors.append(str(exc))
                    print(
                        f"Global fallback ranking failed: candidate from shot "
                        f"{candidate.get('source_slot', '?') + 1}, target shot "
                        f"{slot_index + 1}: {exc}"
                    )
                    continue
                ranked_pairs.append((score, slot_index, candidate, reason))

        assigned_slots = set()
        assigned_candidates = set()
        for score, slot_index, candidate, reason in sorted(
            ranked_pairs, key=lambda item: item[0], reverse=True
        ):
            candidate_key = str(candidate.get("video_id"))
            if slot_index in assigned_slots or candidate_key in assigned_candidates:
                continue
            destination = out_path / f"clip_{slot_index:02d}.mp4"
            candidate["path"].replace(destination)
            downloaded.append(str(destination))
            assigned_slots.add(slot_index)
            assigned_candidates.add(candidate_key)
            used_video_ids.add(candidate["video_id"])
            print(
                f"GLOBAL FALLBACK selected for shot {slot_index + 1}: "
                f"score={score}/100; candidate originally collected for shot "
                f"{candidate.get('source_slot', '?') + 1}; {reason}"
            )

        for candidate in fallback_pool:
            if str(candidate.get("video_id")) not in assigned_candidates:
                candidate["path"].unlink(missing_ok=True)

        still_missing = sorted(set(missing_fallback_slots) - assigned_slots)
        if still_missing:
            raise RuntimeError(
                f"Global fallback pool could not fill {len(still_missing)} "
                f"missing shot(s): {[i + 1 for i in still_missing]}. "
                f"Available rejected clips: {len(fallback_pool)}; "
                f"ranking errors: {ranking_errors[:3]}"
            )

    # Rebuild the result in exact slot order. Fallback clips are assigned
    # after the normal per-slot pass, so appending them directly would shift
    # their narration positions during editing.
    ordered_downloads = [
        str(out_path / f"clip_{slot_index:02d}.mp4")
        for slot_index in range(len(shots))
    ]
    missing_paths = [
        path for path in ordered_downloads
        if not Path(path).exists() or Path(path).stat().st_size <= 10000
    ]
    if missing_paths:
        raise RuntimeError(
            f"Clip sourcing produced {len(missing_paths)} missing/invalid slot file(s): "
            f"{missing_paths[:5]}"
        )
    downloaded = ordered_downloads

    missing = len(shots) - len(downloaded)
    if missing:
        raise RuntimeError(f"Internal error: {missing} shot(s) remain without accepted clips.")

    print(
        f"Cosmic Curious clip sourcing: downloaded {len(downloaded)}/{len(shots)} "
        "clips (up to 3 Visual-QA search rounds per missing slot; "
        "best rejected candidate ranked as fallback; "
        "all 15 species-focused searches; global best-fit fallback across missing slots)."
    )
    return downloaded
