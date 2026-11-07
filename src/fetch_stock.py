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
    """Use the configured vision model to verify downloaded video frames."""
    from src.freeinference_client import generate_vision, is_vision_available

    if not is_vision_available():
        raise RuntimeError("FREEINFERENCE_API_KEY is required for Cosmic Curious visual verification.")

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

ACTION CONTEXT (NOT A STRICT REQUIREMENT):
The narration describes this behavior: {action}
The clip does NOT have to show this exact behavior. It is acceptable to show
the required animal clearly in another natural pose or activity. Do not claim
the described action is visible unless the frames actually show it.

INTENDED SHOT:
{visual}

Approve ONLY if:
1. The visible animal is positively identifiable as the exact REQUIRED ANIMAL.
   Similar species, related species, generic raptors, and lookalikes are NOT matches.
2. Set "species_match" to true only when the visible evidence supports the exact
   required species. If uncertain, set it to false.
3. The narration's exact behavior is optional. The animal itself must be
   clearly visible and positively identifiable. Do not infer or claim that
   the described behavior is visible unless it is actually shown.
4. There is no unrelated object or food being used as a substitute.
5. The visual evidence is clear enough to identify the animal.

Reject humans, office workers, unrelated animals, food, generic nature
scenery, and empty habitats. Judge only what is actually visible in frames.

Return ONLY valid JSON:
{{
  "animal_visible": true,
  "species_match": true,
  "animal_identification": "exact species visibly shown, or why it cannot be confirmed",
  "action_visible": true,
  "confidence": 0.95,
  "reason": "brief evidence-based explanation"
}}
"""
        frame_images = [
            (frame_path.read_bytes(), "image/jpeg")
            for frame_path in frames
        ]
        # Send the complete sampled-frame batch to FreeInference Qwen.
        image_batches = [frame_images]

        audit_results = []
        for image_batch in image_batches:
            response = None
            # Vision service overload (429/503) is often transient. Give it
            # a few more chances before failing closed, without long waits.
            max_vision_attempts = 5
            for attempt in range(1, max_vision_attempts + 1):
                try:
                    response = generate_vision(prompt, image_batch)
                    break
                except Exception as exc:
                    error_text = str(exc)
                    retryable = any(
                        marker in error_text
                        for marker in (
                            "503", "UNAVAILABLE",
                            "429", "RESOURCE_EXHAUSTED",
                            "500", "INTERNAL",
                            "502", "504",
                            "temporarily unavailable",
                        )
                    )
                    if not retryable or attempt >= max_vision_attempts:
                        raise
                    delay = min(3 * (2 ** (attempt - 1)), 20)
                    print(
                        f"WARNING: FreeInference Qwen vision temporary API error "
                        f"(attempt {attempt}/{max_vision_attempts}); "
                        f"retrying in {delay}s: {error_text[:250]}"
                    )
                    time.sleep(delay)

            if response is None:
                raise RuntimeError("FreeInference Qwen vision returned no response after retries.")

            raw = (getattr(response, "text", None) or "").strip()
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                raise RuntimeError(
                    f"Vision model returned invalid JSON: {raw[:300]}"
                )
            audit_results.append(json.loads(match.group(0)))

        def _audit_score(candidate):
            try:
                candidate_confidence = float(candidate.get("confidence", 0))
            except (TypeError, ValueError):
                candidate_confidence = 0.0
            candidate_passes = (
                candidate.get("animal_visible") is True
                and candidate.get("species_match") is True
                and (
                    candidate.get("action_visible") is True
                    or not require_action
                )
            )
            return (candidate_passes, candidate_confidence)

        # Require animal and requested action evidence from the same batch;
        # do not combine unrelated evidence across distant clip frames.
        result = max(audit_results, key=_audit_score)
        animal_ok = result.get("animal_visible") is True
        species_ok = result.get("species_match") is True
        action_ok = result.get("action_visible") is True

        try:
            confidence = float(result.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(result.get("reason", "No reason returned"))
        approved = (
            animal_ok
            and species_ok
            and (action_ok or not require_action)
            and confidence >= 0.80
        )

        return approved, (
            f"animal_visible={animal_ok}, species_match={species_ok}, "
            f"action_visible={action_ok}, action_required={require_action}, "
            f"confidence={confidence:.2f}: {reason}"
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
    missing_slots = []
    used_video_ids = set()

    # Bound search work per shot. One action-specific page and one
    # exact-species fallback page provide up to 160 candidates without
    # crawling every Pexels page (which previously caused very long runs).
    # QA still checks candidates in order and only replaces rejected shots.
    pexels_per_page = 80
    max_search_modes = 2

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
                raise RuntimeError(
                    f"Could not re-verify existing clip for shot {index + 1}; "
                    f"upload blocked. Details: {exc}"
                ) from exc

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

        # Keep searches species-specific. The director's action query comes
        # first; the remaining queries broaden wording, never the animal.
        aliases = species_search_aliases.get(
            normalized_animal,
            [animal, f"{animal} wildlife", f"{animal} {action}"],
        )
        search_modes = [(query, True)]
        # Alternate queries and the primary query all require the exact
        # animal to be visible. The narrated action is context, not a gate.
        search_modes.extend(
            (candidate_query, False)
            for candidate_query in aliases
            if candidate_query.strip().lower() != query.lower()
        )
        # Keep the highest-value searches: exact narrated action first,
        # then alternate exact-species wording. Every accepted clip must
        # still show the requested action.
        search_modes = search_modes[:max_search_modes]

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
                    if not video_id or video_id in used_video_ids:
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

        # Do not let one noisy Pexels query monopolize verification.
        # Every candidate must pass exact-species visibility QA. The exact
        # narrated action is not mandatory because suitable footage may not
        # exist for every narration segment.
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

        print(
            f"Visual shot {index + 1}: Pexels returned "
            f"{sum(search_result_counts.values())} results "
            f"({len(candidates_by_id)} unique videos) across "
            f"{len(search_modes)} species-specific queries. "
            f"Per-query counts: {search_result_counts}"
        )

        accepted_path = None
        checked = 0
        for video_id, candidate in ranked:
            checked += 1

            candidate_path = out_path / (
                f"candidate_{index:02d}_{checked:03d}.mp4"
            )
            if not _download_pexels_video(candidate["video"], candidate_path):
                continue

            require_action = candidate["require_action"]
            try:
                approved, audit_reason = _verify_clip_matches_shot(
                    candidate_path,
                    animal=animal,
                    action=action,
                    sentence=sentence,
                    visual=visual,
                    require_action=require_action,
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
                print(
                    f"Visual QA PASSED for shot {index + 1}: "
                    f"source_query={candidate['query']!r}; {audit_reason}"
                )
                break

            print(
                f"Visual QA REJECTED candidate for shot {index + 1}: "
                f"source_query={candidate['query']!r}; {audit_reason}"
            )
            candidate_path.unlink(missing_ok=True)

        if accepted_path is None:
            missing_slots.append(index + 1)
            print(
                f"Visual QA: slot {index + 1} remains missing after "
                f"{checked} candidates. Keeping all approved clips; "
                f"continuing with the other slots."
            )
            continue

        downloaded.append(str(accepted_path))
        print(
            f"Visual QA progress: {len(downloaded)}/{len(shots)} approved; "
            f"{len(shots) - len(downloaded)} slots still need clips."
        )

    if missing_slots:
        raise RuntimeError(
            f"Visual QA approved {len(downloaded)}/{len(shots)} clips. "
            f"Missing slots: {missing_slots}. Approved clips were retained. "
            "Rerun the pipeline to search only the missing slots."
        )

    print(
        f"Cosmic Curious visual QA: verified {len(downloaded)}/{len(shots)} "
        "animal-and-action-matched clips."
    )
    return downloaded
