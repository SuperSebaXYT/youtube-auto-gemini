from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str, shot_count: int = 15, title: str = "") -> dict:
    """Create an ordered, narration-synced shot plan for one animal video."""
    if shot_count < 1:
        raise ValueError("shot_count must be at least 1")

    return run_json_batch(
        "Visual Director",
        """Create a precise, narration-synced wildlife shot list.

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

Return exactly the requested number of shots, in narration order. Each shot
must cover a distinct consecutive part of the narration. Do not skip or
reorder narration.""",
        f'''Create exactly {shot_count} consecutive shots for this video.

TOPIC: {topic}
FULL NARRATION:
{script}

For every shot return:
- "sentence": the exact sentence or short consecutive narration segment
- "animal": exact common animal/species name visible in the shot
- "action": exact visible behavior/action that must be shown
- "visual": a concrete description of what the camera must show
- "search_query": concise English Pexels query containing the animal and action

Reject any idea that would show only an object, food, scenery, or an unrelated
animal. The animal and action must both be explicit in every shot.

Return exactly:
{{"shots":[{{"sentence":"...","animal":"...","action":"...","visual":"...","search_query":"..."}}]}}.''',
        max_output_tokens=4500,
    )
