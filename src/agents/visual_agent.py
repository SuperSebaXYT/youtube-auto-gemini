from __future__ import annotations

import json

from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str, shot_count: int = 15, title: str = "") -> dict:
    """Generate exactly one shot per narration slot, retrying invalid plans."""
    if shot_count < 1:
        raise ValueError("shot_count must be at least 1")

    role = """You are a strict wildlife footage director.

Every shot must name the exact animal species relevant to its narration
segment. The animal must be the main visible subject. Search queries must
be concise English phrases naming that exact animal and, where practical,
its natural behavior. Never use generic wildlife, animal, nature, or
documentary filler queries.

Do not invent species or behavior. If an exact narrated action is unlikely
to be filmable, describe the same species in a clearly relevant natural
pose or activity. No unrelated animals, scenery-only shots, objects,
people, food, diagrams, or metaphors.

Narration segments must be exact consecutive words copied from the script.
Together, the segments must cover the full narration in order, without
missing, repeated, or rearranged words."""

    base_task = f'''Create exactly {shot_count} visual shot specifications for this video.
Create ONE shot per narration slot. Do not create alternatives, candidate
pools, or extra shots. If footage later fails Visual QA, the retrieval loop
will search only for that missing slot.

TITLE: {title}
TOPIC: {topic}
FULL NARRATION:
{script}

Divide the complete narration into exactly {shot_count} consecutive,
chronological segments. Each "sentence" must be copied verbatim. Together,
all segments must cover the entire narration in order.

For each slot include:
- "slot": integer 1 through {shot_count}, exactly once and in order
- "sentence": exact consecutive narration words
- "animal": exact common animal/species name
- "action": relevant natural behavior (context, not a mandatory footage gate)
- "visual": concrete description showing that animal clearly
- "search_query": concise, specific English Pexels query naming that animal

Return exactly {shot_count} objects and no alternatives.
Return only:
{{"shots":[{{"slot":1,"sentence":"...","animal":"...","action":"...","visual":"...","search_query":"..."}}]}}'''

    def shots_are_valid(items: list) -> bool:
        if len(items) != shot_count or not all(isinstance(x, dict) for x in items):
            return False
        if [x.get("slot") for x in items] != list(range(1, shot_count + 1)):
            return False
        fields = ("sentence", "animal", "action", "visual", "search_query")
        return all(
            isinstance(item.get(field), str) and item[field].strip()
            for item in items
            for field in fields
        )

    max_attempts = 5
    previous_result = None
    last_problem = "No valid shot plan was returned."

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            task = base_task
        else:
            task = f'''The previous shot plan was invalid. Correct it completely.
This is attempt {attempt} of {max_attempts}. Return a FULL replacement plan,
not a partial patch. The replacement must contain exactly {shot_count} shots,
with slot numbers 1 through {shot_count} exactly once and in order.

VALIDATION FAILURE:
{last_problem}

ORIGINAL REQUIREMENTS:
{base_task}

PREVIOUS INVALID RESULT:
{json.dumps(previous_result, ensure_ascii=False) if previous_result is not None else "The previous response could not be parsed as JSON."}

Do not repeat the same structural mistake. Return only the complete JSON object.'''

        print(
            f"Visual Director: generating shot plan "
            f"(attempt {attempt}/{max_attempts})."
        )

        try:
            generated = run_json_batch(
                "Visual Shot Planner",
                role,
                task,
                max_output_tokens=7000,
            )
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            previous_result = None
            last_problem = f"Response was not valid usable JSON: {str(exc)[:500]}"
            print(f"Visual Director: invalid JSON on attempt {attempt}: {last_problem}")
            continue

        shots = generated.get("shots", []) if isinstance(generated, dict) else []
        if shots_are_valid(shots):
            print(
                f"Visual Director: created {shot_count} shot specifications "
                f"after {attempt} attempt(s) (one per slot; no pre-generated alternatives)."
            )
            return {"shots": shots}

        previous_result = generated
        if not isinstance(generated, dict):
            last_problem = "The response root must be a JSON object containing a 'shots' array."
        elif not isinstance(generated.get("shots"), list):
            last_problem = "The response must contain a 'shots' array."
        elif len(shots) != shot_count:
            last_problem = (
                f"Received {len(shots)} shots; exactly {shot_count} are required."
            )
        elif [x.get("slot") for x in shots if isinstance(x, dict)] != list(range(1, shot_count + 1)):
            last_problem = (
                f"Slot numbers must be exactly 1 through {shot_count}, once each, in order."
            )
        else:
            last_problem = (
                "Every shot must be an object with non-empty sentence, animal, "
                "action, visual, and search_query string fields."
            )

        print(f"Visual Director: rejected shot plan on attempt {attempt}: {last_problem}")

    raise RuntimeError(
        f"Visual Director failed to produce exactly {shot_count} valid shots "
        f"after {max_attempts} attempts. Last validation error: {last_problem}"
    )
