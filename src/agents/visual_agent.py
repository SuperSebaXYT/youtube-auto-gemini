from __future__ import annotations

import re

from src.agents.orchestrator import run_json_batch


# Used only to complete a short AI response or recover from an invalid one.
# These are deliberately camera/framing variations, so they remain usable
# for species whose movement differs (for example, fish versus mammals).
FALLBACK_VIEWS = [
    ("natural movement", "wide shot showing the whole animal clearly", "wide shot"),
    ("natural movement", "full body side profile", "side view"),
    ("natural movement", "full body viewed from the front", "front view"),
    ("natural movement", "close-up of the head", "head close up"),
    ("natural movement", "close-up of the eyes and face", "face close up"),
    ("natural movement", "close-up showing the animal's body markings", "body markings close up"),
    ("natural movement", "the animal clearly visible in its natural habitat", "natural habitat wide shot"),
    ("natural movement", "medium shot from a low camera angle", "low angle view"),
    ("natural movement", "medium shot from a slightly elevated camera angle", "elevated angle view"),
    ("natural movement", "view from above with the animal clearly visible", "overhead view"),
    ("resting naturally", "the animal resting, viewed from the side", "resting side view"),
    ("moving naturally", "the animal moving toward the camera", "moving toward camera"),
    ("moving naturally", "the animal moving away from the camera", "moving away camera"),
    ("feeding naturally when appropriate", "clear medium shot of the animal in its environment", "feeding natural behavior"),
    ("natural movement", "a different close-up angle showing the full head and body", "alternate close up angle"),
]


def _clean(value: object, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _build_fallback_shots(animal: str, shot_count: int) -> list[dict]:
    shots = []
    for index in range(shot_count):
        action, visual, angle = FALLBACK_VIEWS[index % len(FALLBACK_VIEWS)]
        query = f"{animal} {angle}"
        if index >= len(FALLBACK_VIEWS):
            query += f" different footage angle {index + 1}"
        shots.append({
            "slot": index + 1,
            "sentence": f"General visual coverage of {animal}; not tied to a narration segment.",
            "animal": animal,
            "action": action,
            "visual": visual,
            "search_query": query,
        })
    return shots


def plan(topic: str, script: str, shot_count: int = 15, title: str = "") -> dict:
    """Create species-focused search variations, not narration-aligned scenes."""
    if shot_count < 1:
        raise ValueError("shot_count must be at least 1")

    topic_name = _clean(topic, 120) or _clean(title, 120) or "wild animal"

    role = """You are a wildlife stock-footage SEARCH PLANNER, not an image or video generator.
Your job is to create varied search phrases for finding REAL existing footage.

Use one exact target animal species for every shot. Do not switch species.
Suggest different natural activities when they genuinely fit this species, such
as walking, swimming, resting, feeding, jumping, flying, or moving through its
habitat. Never assign an impossible activity to the species. Mix these with
different camera perspectives: side, front, wide, close-up, overhead, low angle,
and alternate views of a similar activity.

Do not divide, summarize, quote, or align the narration. Do not make each shot
represent a sentence or fact. Repeating an activity is allowed when the camera
angle or framing is different. Every search phrase must explicitly name the
same target species. Do not request generated, animated, illustrated, or AI
made footage. Search only for real wildlife video clips."""

    task = f'''Create a varied visual search plan for an existing-footage wildlife video.

VIDEO TITLE: {title}
TARGET TOPIC / SPECIES: {topic_name}

The narration is provided only as background and MUST NOT be divided into
segments or used to assign one shot per sentence:
{_clean(script, 12000)}

Return exactly {shot_count} shot ideas if possible. Prioritize different
natural activities and camera perspectives. It is acceptable to repeat an
activity with a different camera angle, because the footage search must still
find a separate real clip.

Return JSON only, in this format:
{{"animal":"exact common species name","shots":[{{"slot":1,"action":"...","visual":"...","search_query":"exact species + specific footage search"}}]}}'''

    generated = None
    try:
        generated = run_json_batch(
            "Wildlife Footage Search Planner",
            role,
            task,
            max_output_tokens=5000,
        )
    except Exception as exc:
        print(
            "Visual Director: AI search-plan request failed; "
            f"using built-in species-focused camera variations: {exc}"
        )

    animal = topic_name
    raw_shots = []
    if isinstance(generated, dict):
        animal = _clean(generated.get("animal"), 120) or topic_name
        candidate_shots = generated.get("shots")
        if isinstance(candidate_shots, list):
            raw_shots = candidate_shots

    shots = []
    seen_queries = set()
    for item in raw_shots:
        if not isinstance(item, dict):
            continue
        action = _clean(item.get("action"), 120) or "moving naturally"
        visual = _clean(item.get("visual"), 180) or "the animal clearly visible"
        query = _clean(item.get("search_query"), 160)
        if not query:
            query = f"{animal} {visual}"
        elif animal.casefold() not in query.casefold():
            query = f"{animal} {query}"
        normalized = re.sub(r"[^a-z0-9]+", " ", query.casefold()).strip()
        if not normalized or normalized in seen_queries:
            continue
        seen_queries.add(normalized)
        shots.append({
            "slot": len(shots) + 1,
            "sentence": f"General visual coverage of {animal}; not tied to a narration segment.",
            "animal": animal,
            "action": action,
            "visual": visual,
            "search_query": query[:180],
        })
        if len(shots) >= shot_count:
            break

    # A language model may return fewer shots than requested. Complete the
    # plan locally instead of failing the entire video because it returned 9.
    if len(shots) < shot_count:
        for fallback in _build_fallback_shots(animal, shot_count):
            normalized = re.sub(
                r"[^a-z0-9]+", " ", fallback["search_query"].casefold()
            ).strip()
            if normalized in seen_queries:
                fallback["search_query"] += f" alternate camera perspective {len(shots) + 1}"
                normalized = re.sub(
                    r"[^a-z0-9]+", " ", fallback["search_query"].casefold()
                ).strip()
            seen_queries.add(normalized)
            fallback["slot"] = len(shots) + 1
            shots.append(fallback)
            if len(shots) >= shot_count:
                break

    print(
        f"Visual Director: prepared {len(shots)} species-focused search shots "
        f"for {animal!r}; narration was not split into shot segments."
    )
    return {"shots": shots}
