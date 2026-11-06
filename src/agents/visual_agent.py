from __future__ import annotations

import json

from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str, shot_count: int = 15, title: str = "") -> dict:
    """Generate 3 visual candidates per narration segment, then select the best 15."""
    if shot_count < 1:
        raise ValueError("shot_count must be at least 1")

    candidate_count = shot_count * 3

    role = """You are a strict wildlife footage director and visual fact-checker.

NON-NEGOTIABLE SPECIES MATCH:
Every proposed shot must visibly show the exact animal species named or
clearly implied by its narration segment. The animal must be the main subject.

ACTION MATCH:
The visible behavior must match the narration. If it describes spraying,
striking, stinging, hunting, building, swimming, flying, or another action,
the proposed footage must show that action in progress. A generic animal
close-up is not an adequate substitute for a specific narrated behavior.

ABSOLUTE BANS:
No food, cooking, kitchens, objects, tools, products, people, diagrams,
metaphors, unrelated animals, generic nature filler, or scenery-only shots.
Habitat is allowed only as background behind the correct animal.

Never invent evidence. If an exact behavior is unlikely to be filmable,
describe the same species performing its closest directly relevant natural
behavior, and make that limitation explicit.

Search queries must be concise English phrases naming the exact animal and
visible behavior. Disambiguate animal names that also mean objects or jobs.
Never use generic queries such as wildlife, animals, nature, or documentary.

Narration segments must be exact consecutive words copied from the script.
Do not paraphrase, omit, repeat, or reorder narration."""

    # Stage 1: create three genuinely different footage options for each
    # of the 15 consecutive narration segments (45 candidates total).
    candidate_task = f'''Generate exactly {candidate_count} visual candidates for
this video: exactly 3 alternatives for each of {shot_count} narration slots.

TITLE: {title}
TOPIC: {topic}
FULL NARRATION:
{script}

First divide the complete narration into exactly {shot_count} consecutive,
chronological slots. Each slot should cover a meaningful short segment.
The "sentence" value must be copied verbatim from the narration. Together,
the 15 slot sentences must cover the entire narration in order, with no
missing or repeated words.

For EACH slot, create exactly 3 different candidate shots:
- Keep all 3 alternatives tied to that slot's exact narration segment.
- Prefer different realistic camera views or directly relevant behaviors.
- Do not change the animal species to create variety.
- Do not invent behavior that the narration does not support.
- Each candidate must have a distinct search query.

Return exactly {candidate_count} objects. Use candidate_id values 1 through
{candidate_count}, and slot values 1 through {shot_count}. Each slot must
have exactly three candidates. Include:
- "candidate_id": unique integer from 1 to {candidate_count}
- "slot": narration slot integer from 1 to {shot_count}
- "sentence": exact consecutive narration words for this slot
- "animal": exact common animal/species name
- "action": exact visible behavior required
- "visual": concrete description of the footage
- "search_query": concise English Pexels query naming animal and action

No candidate may be scenery, an object, food, or an unrelated animal.
Return only:
{{"candidates":[{{"candidate_id":1,"slot":1,"sentence":"...","animal":"...","action":"...","visual":"...","search_query":"..."}}]}}'''

    generated = run_json_batch(
        "Visual Candidate Generator",
        role,
        candidate_task,
        max_output_tokens=11000,
    )
    candidates = (
        generated.get("candidates", [])
        if isinstance(generated, dict)
        else []
    )

    def candidates_are_valid(items: list) -> bool:
        if len(items) != candidate_count:
            return False
        ids = [item.get("candidate_id") for item in items if isinstance(item, dict)]
        slots = [item.get("slot") for item in items if isinstance(item, dict)]
        if len(ids) != candidate_count or len(set(ids)) != candidate_count:
            return False
        if set(ids) != set(range(1, candidate_count + 1)):
            return False
        if not all(
            isinstance(item.get(field), str) and item[field].strip()
            for item in items
            for field in ("sentence", "animal", "action", "visual", "search_query")
        ):
            return False
        return all(slots.count(slot) == 3 for slot in range(1, shot_count + 1))

    if not candidates_are_valid(candidates):
        repair_task = f'''Repair this invalid candidate list. Return exactly
{candidate_count} candidates, with IDs 1..{candidate_count} and exactly
three alternatives for each slot 1..{shot_count}. Preserve exact narration
segments, species accuracy, and action accuracy. Do not invent footage.

NARRATION:
{script}

INVALID RESULT:
{json.dumps(generated, ensure_ascii=False)}

Return only JSON using the required "candidates" schema.'''
        generated = run_json_batch(
            "Visual Candidate Generator",
            role,
            repair_task,
            max_output_tokens=11000,
        )
        candidates = (
            generated.get("candidates", [])
            if isinstance(generated, dict)
            else []
        )

    if not candidates_are_valid(candidates):
        raise RuntimeError(
            f"Visual Director generated an invalid candidate pool "
            f"({len(candidates)} received; {candidate_count} required, "
            "three per narration slot)."
        )

    # Stage 2: OpenCode judges all 45 options and selects exactly one best
    # species/action-matched candidate for each narration slot.
    selection_task = f'''Select the best {shot_count} shots from this pool of
{candidate_count} candidates. You MUST select exactly one candidate for
each narration slot 1..{shot_count}, preserving slot order.

Selection priority:
1. Exact animal species match.
2. Exact visible action match to the narration (not merely a similar animal).
3. Footage that is realistically searchable and filmable.
4. Clear visual composition with the animal as the main subject.
5. Search query that is specific and unambiguous.

Reject candidates that show generic animal footage, scenery, objects, or
actions unsupported by the narration. Do not choose multiple candidates
from one slot. Do not invent or rewrite candidate details.

NARRATION:
{script}

ALL {candidate_count} CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

Return exactly {shot_count} selected objects in slot order, preserving the
selected candidate's fields and adding "selected_candidate_id". The slots
must be exactly 1..{shot_count}, one each.
Return only:
{{"shots":[{{"selected_candidate_id":1,"slot":1,"sentence":"...","animal":"...","action":"...","visual":"...","search_query":"..."}}]}}'''

    selected = run_json_batch(
        "Visual Director Selector",
        role,
        selection_task,
        max_output_tokens=7000,
    )
    shots = selected.get("shots", []) if isinstance(selected, dict) else []

    def selection_is_valid(items: list) -> bool:
        if len(items) != shot_count:
            return False
        slots = [item.get("slot") for item in items if isinstance(item, dict)]
        ids = [item.get("selected_candidate_id") for item in items if isinstance(item, dict)]
        if len(slots) != shot_count or len(ids) != shot_count:
            return False
        if slots != list(range(1, shot_count + 1)):
            return False
        candidate_by_id = {item["candidate_id"]: item for item in candidates}
        if len(set(ids)) != shot_count or any(cid not in candidate_by_id for cid in ids):
            return False
        required_fields = (
            "slot",
            "sentence",
            "animal",
            "action",
            "visual",
            "search_query",
        )
        for shot in items:
            source = candidate_by_id[shot["selected_candidate_id"]]
            if source["slot"] != shot["slot"]:
                return False
            # The selector may choose a candidate, but it must not silently
            # rewrite its species, action, narration segment or search query.
            if any(shot.get(key) != source.get(key) for key in required_fields):
                return False
        return True

    if not selection_is_valid(shots):
        correction_task = f'''Your selection is invalid. Select exactly one
candidate for every slot 1..{shot_count}, in ascending slot order. Use only
candidate IDs from the supplied pool. Do not modify candidate details.

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False)}

INVALID SELECTION:
{json.dumps(selected, ensure_ascii=False)}

Return exactly {shot_count} shots with fields selected_candidate_id, slot,
sentence, animal, action, visual, and search_query.'''
        selected = run_json_batch(
            "Visual Director Selector",
            role,
            correction_task,
            max_output_tokens=7000,
        )
        shots = selected.get("shots", []) if isinstance(selected, dict) else []

    if not selection_is_valid(shots):
        raise RuntimeError(
            f"OpenCode could not select exactly {shot_count} valid shots "
            f"from the {candidate_count}-candidate pool."
        )

    print(
        f"Visual Director: OpenCode generated {candidate_count} candidates "
        f"and selected {shot_count} narration-matched shots."
    )
    return {"shots": shots}
