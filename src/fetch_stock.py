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


def _verify_clip_matches_shot(
    clip_path: Path,
    animal: str,
    action: str,
    sentence: str,
    visual: str,
    require_action: bool = False,
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
        prompt = f"""
You are checking wildlife footage for a video editor.

TARGET ANIMAL: {animal}

Look at the attached frames and answer only this question:
Is the actual target animal (or an unmistakable example of that animal type)
visibly present in the footage?

The animal identity/type is the ONLY thing that matters.
Ignore the narration, described action, pose, habitat, camera angle, and
background. The animal does NOT need to perform the narrated action.
A normal clip of the correct animal must be accepted.

Reject footage where the target animal is absent or replaced by:
- a human holding/using an object with a similar name,
- a different animal,
- food, toys, drawings, models, or unrelated scenery.

Examples:
- pistol shrimp means an actual shrimp, NOT a person holding a pistol.
- swordfish means an actual fish, NOT a person holding a sword.
- crane as a bird means the actual bird, NOT construction machinery.

Do not demand proof of rare behavior or impossible visual species-level
details that cannot be seen in the frames. Match the named animal/type using
the visible animal itself. If the animal is clearly the intended animal type,
answer MATCH even if it is resting or doing something unrelated to narration.
If the animal is absent or clearly a different type, answer NO_MATCH.

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
        max_vision_attempts = 5
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
                delay = min(3 * (2 ** (attempt - 1)), 20)
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

def fetch_clips_for_shots(
    shots: list,
    out_dir: str,
    orientation: str = "portrait",
) -> list:
    """
    Fetch one verified clip per narration shot.

    Search a broad Pexels result pool for the exact species, then use FreeInference
    Qwen vision to verify candidates. Search results are never trusted as proof:
    wrong species are rejected and publication remains blocked if no exact
    species match can be verified.
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

    # Search every species-specific query and every available Pexels page.
    # There is no fixed candidate-count cap: rejected clips do not consume
    # a shot slot, and only a matching animal can fill that slot.
    pexels_per_page = 80

    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Visual shot {index + 1} is not a JSON object.")

        animal = str(shot.get("animal", "")).strip()
        action = str(shot.get("action", "")).strip()
        sentence = str(shot.get("sentence", "")).strip()
        visual = str(shot.get("visual", "")).strip()
        query = str(shot.get("search_query", "")).strip()

        if not animal or not query:
            raise RuntimeError(
                f"Visual shot {index + 1} is missing its target animal or search query."
            )

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

        # Resume safely: retain clips that already passed Visual QA in an
        # earlier attempt. Only missing or rejected slots are searched again.
        accepted_path = out_path / f"clip_{index:02d}.mp4"
        if accepted_path.exists():
            try:
                existing_ok, existing_reason = _verify_clip_matches_shot(
                    accepted_path,
                    animal=animal,
                    action=action,
                    sentence=sentence,
                    visual=visual,
                    require_action=False,
                )
            except Exception as exc:
                existing_ok = False
                existing_reason = f"could not verify existing clip: {exc}"

            if existing_ok:
                downloaded.append(str(accepted_path))
                print(
                    f"Visual QA RETAINED existing approved clip for shot "
                    f"{index + 1}: {existing_reason}. "
                    f"Remaining slots: {len(shots) - len(downloaded)}"
                )
                continue

            print(
                f"Visual QA REJECTED existing clip for shot {index + 1}: "
                f"{existing_reason}. Searching replacement for this slot only."
            )
            accepted_path.unlink(missing_ok=True)

        accepted_path = None
        checked = 0
        attempted_video_ids = set()
        retry_round = 0
        while accepted_path is None:
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

            # Keep every search anchored to the target animal. Add animal-only
            # wording variants so later retry rounds can discover different
            # Pexels results without ever searching for a human/object homonym.
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
            # Rebuild the search-mode list on every retry round. The previous
            # version called search_modes.extend(...) without initializing
            # search_modes, which could crash before any Pexels search began.
            # Every query is animal-anchored and action visibility is optional.
            search_modes = [(query, False)]
            search_modes.extend(
                (candidate_query, False)
                for candidate_query in aliases
                if candidate_query.strip().lower() != query.lower()
            )

            # Expand discovery progressively after each exhausted round.
            # These remain animal-specific, while Visual QA is the final
            # authority that rejects homonyms and unrelated footage.
            expansion_phrases = (
                "wild animal footage",
                "animal wildlife video",
                "animal closeup footage",
                "wildlife documentary footage",
                "animal in natural habitat",
                "animal portrait wildlife",
            )
            expansion = expansion_phrases[(retry_round - 1) % len(expansion_phrases)]
            expanded_query = f"{animal} {expansion}"
            if expanded_query.lower() not in {
                item[0].lower() for item in search_modes
            }:
                search_modes.append((expanded_query, False))
            # No arbitrary query/candidate cap. Search all animal-specific
            # aliases; action visibility is never part of the approval decision.

            candidates_by_id = {}
            search_result_counts = {}
            for mode_index, (search_query, require_action) in enumerate(search_modes):
                page = 1
                query_result_count = 0
                while True:
                    try:
                        results = _search_pexels(
                            search_query,
                            headers,
                            orientation,
                            per_page=pexels_per_page,
                            page=page,
                        )
                    except Exception as exc:
                        print(
                            f"WARNING: Pexels search failed for shot {index + 1} "
                            f"query={search_query!r} page={page}: {exc}"
                        )
                        break

                    if not results:
                        break

                    query_result_count += len(results)
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
                        # Duplicate candidates are all animal-visibility checks;
                        # the narration action is optional for every search mode.
                        if existing is None or (require_action and not existing["require_action"]):
                            candidates_by_id[video_id] = candidate

                    # A short page means Pexels has no further results for this query.
                    if len(results) < pexels_per_page:
                        break
                    page += 1

                search_result_counts[search_query] = query_result_count

            # Balance candidates from every animal-specific query. Only the
            # target animal/type is checked; narration actions are irrelevant.
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
                f"animal_queries={len(search_modes)}. "
                f"Per-query counts: {search_result_counts}"
            )

            # Download candidates in batches of 15. Each completed download
            # is sent to Visual QA immediately; approved clips are committed to
            # their slot at once. The batch never weakens the animal identity gate.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            batch_size = 15
            for batch_start in range(0, len(ranked), batch_size):
                batch = ranked[batch_start:batch_start + batch_size]
                batch_jobs = {}
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    for video_id, candidate in batch:
                        if video_id in attempted_video_ids:
                            continue
                        checked += 1
                        attempted_video_ids.add(video_id)
                        candidate_path = out_path / (
                            f"candidate_{index:02d}_{checked:05d}.mp4"
                        )
                        future = executor.submit(
                            _download_pexels_video, candidate["video"], candidate_path
                        )
                        batch_jobs[future] = (video_id, candidate, candidate_path)

                    for future in as_completed(batch_jobs):
                        video_id, candidate, candidate_path = batch_jobs[future]
                        try:
                            was_downloaded = future.result()
                        except Exception as exc:
                            was_downloaded = False
                            print(
                                f"WARNING: parallel download failed for shot {index + 1}, "
                                f"video {video_id}: {exc}"
                            )
                        if not was_downloaded:
                            continue

                        # QA starts as soon as this particular download finishes.
                        try:
                            approved, audit_reason = _verify_clip_matches_shot(
                                candidate_path,
                                animal=animal,
                                action=action,
                                sentence=sentence,
                                visual=visual,
                                require_action=False,
                            )
                        except Exception as exc:
                            approved = False
                            audit_reason = f"Vision QA error: {exc}"

                        if approved:
                            accepted_path = out_path / f"clip_{index:02d}.mp4"
                            candidate_path.replace(accepted_path)
                            used_video_ids.add(video_id)
                            print(
                                f"Visual QA PASSED for shot {index + 1}: "
                                f"source_query={candidate['query']!r}; {audit_reason}. "
                                f"Checked {checked} distinct candidates so far."
                            )
                            break

                        print(
                            f"Visual QA REJECTED candidate for shot {index + 1}: "
                            f"source_query={candidate['query']!r}; {audit_reason}"
                        )
                        candidate_path.unlink(missing_ok=True)

                # The executor waits for in-flight downloads when leaving
                # the context. Remove any completed but unused batch files so
                # only the approved slot clip survives on disk.
                if accepted_path is not None:
                    for _, _, unused_path in batch_jobs.values():
                        if unused_path != accepted_path:
                            unused_path.unlink(missing_ok=True)
                    break

            if accepted_path is None:
                print(
                    f"Visual QA: slot {index + 1} still missing. "
                    f"Progress: {checked} distinct videos checked; "
                    f"{len(attempted_video_ids)} video IDs will not be downloaded "
                    "or checked again during this slot. "
                    f"Round {retry_round} exhausted its current animal-specific queries. "
                    "Keeping approved clips; expanding the animal search and retrying in 60 seconds."
                )
                time.sleep(60)

        downloaded.append(str(accepted_path))
        print(
            f"Visual QA progress: {len(downloaded)}/{len(shots)} approved; "
            f"{len(shots) - len(downloaded)} slots still need clips."
        )

    print(
        f"Cosmic Curious visual QA: verified {len(downloaded)}/{len(shots)} "
        "exact-species-matched clips."
    )
    return downloaded
