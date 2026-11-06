from __future__ import annotations

import json

from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str, shot_count: int = 15, title: str = "") -> dict:
    """Create an ordered, narration-synced shot plan for one animal video."""
    if shot_count < 1:
        raise ValueError("shot_count must be at least 1")

    role = """Create a precise, narration-synced wildlife shot list.

NON-NEGOTIABLE VISUAL RULE:
Every shot must visibly show the exact animal species being discussed in
that narration segment. The animal must be the main subject, not a tiny
background detail.

ACTION LOCK:
The visible action must match the sentence. If the narration says an animal
sprays, fires, spits, stings, attacks, builds, hunts, swims, flies, or performs
another specific behavior, the shot must show that exact behavior in progress.
A generic close-up of the animal is NOT an acceptable substitute when a
specific action is described.

EXAMPLES:
- "The bombardier beetle fires a boiling chemical spray" means a real
  bombardier beetle visibly discharging its defensive spray. Not pasta,
  boiling liquid, food, a kitchen, or a beetle merely sitting still.
- "Bees spray/defend using a toxin" means the exact bee species and the
  described defensive behavior, not honey, flowers, or generic insects.

ABSOLUTE BANS:
No food, meals, soup, broth, cooking, kitchens, plates, bowls, cutlery,
ingredients, vegetables, fruit, vehicles, bicycles, tools, products, people,
household objects, diagrams, metaphors, unrelated animals, or generic
nature filler. A habitat may appear only as background behind the correct
animal. Never substitute an object for an animal or its behavior.

If the exact behavior is not realistically filmable, use the same species
performing its closest directly relevant natural behavior and state that
honestly in the visual field. Never invent an action or fake evidence.

Search queries must be concise English searches naming the exact animal
and the exact visible behavior. Disambiguate species names that also mean
people or objects. For the secretary bird, use "Sagittarius serpentarius"
or "secretary bird animal"; never use "secretary" by itself and never show
a human office secretary. For snakes, explicitly request a real snake.
Do not include ambiguous words that can lead to unrelated objects. Do not
use generic queries such as "wildlife", "animals", "nature", or "documentary".

SHOT COUNT AND NARRATION COVERAGE:
Return EXACTLY the requested number of shots, even when the narration has
fewer sentences. A shot is a short consecutive time segment, NOT necessarily
a whole sentence. Split long sentences into multiple smaller consecutive
segments. For a 15-shot Short, aim for roughly 5-8 spoken words per segment,
adjusting boundaries to keep each segment meaningful. Cover the narration
from its first word to its last word in order, with no skipped or repeated
words. The "sentence" field must contain the exact consecutive words from
the narration assigned to that shot. Do not invent or paraphrase narration.
Each shot must cover a distinct consecutive segment. Do not combine distant
parts of the narration or reorder anything."""

    task = f'''Create exactly {shot_count} consecutive shots for this video.

TITLE: {title}
TOPIC: {topic}
FULL NARRATION:
{script}

For every shot return:
- "sentence": exact consecutive words copied from the narration for this shot
- "animal": exact common animal/species name visible in the shot
- "action": exact visible behavior/action that must be shown
- "visual": concrete description of what the camera must show
- "search_query": concise English Pexels query containing the animal and action

The narration may have fewer sentences than the requested shot count. Split
sentences into short word-level segments. Do NOT return one shot per sentence.
For {shot_count} shots, return {shot_count} objects. Count them before answering.
Reject any idea that would show only an object, food, scenery, or an unrelated
animal. The animal and action must both be explicit in every shot.

Return exactly:
{{"shots":[{{"sentence":"...","animal":"...","action":"...","visual":"...","search_query":"..."}}]}}.'''

    result = run_json_batch(
        "Visual Director",
        role,
        task,
        max_output_tokens=6500,
    )

    shots = result.get("shots", []) if isinstance(result, dict) else []
    if len(shots) == shot_count:
        return result

    # A count correction is cheaper and safer than silently filling gaps with
    # generic stock footage. Preserve the first response as context and ask
    # Gemini to repartition the same narration into the exact required count.
    correction = f'''Your previous response returned {len(shots)} shots, but exactly
{shot_count} are required. This is invalid.

Rebuild the complete shot list from scratch with EXACTLY {shot_count} shots.
Do not merely return the previous short list. Split the narration into smaller
consecutive word-level segments, roughly 5-8 spoken words per shot.
The "sentence" fields together must cover every narration word exactly once,
in original order, without omissions, repetitions, or paraphrasing.
Keep the exact-species and exact-visible-action rules from your role.

NARRATION:
{script}

INVALID PREVIOUS RESPONSE (for diagnosis only):
{json.dumps(result, ensure_ascii=False)}

Return only JSON with exactly {shot_count} objects in "shots".'''

    corrected = run_json_batch(
        "Visual Director",
        role,
        correction,
        max_output_tokens=6500,
    )
    corrected_shots = (
        corrected.get("shots", [])
        if isinstance(corrected, dict)
        else []
    )
    if len(corrected_shots) != shot_count:
        raise RuntimeError(
            f"Visual Director returned {len(corrected_shots)} shots after "
            f"count correction; exactly {shot_count} are required."
        )

    return corrected
