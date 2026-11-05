from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str) -> dict:
    return run_json_batch(
        "Visual Director",
        """Create scene-specific visuals that match the narration exactly.
The channel is exclusively about real animals and wildlife.

HARD VISUAL RULE: EVERY shot must visibly feature the exact animal species
being discussed. The animal must be the clear main subject and occupy most
of the frame. If a sentence cannot be illustrated with that animal, show
that same animal in a relevant natural behavior or habitat.

ABSOLUTE BANS: Never show food, meals, soup, broth, cooking, kitchens,
plates, bowls, cutlery, ingredients, vegetables, fruit, or food preparation.
Never show bicycles, bikes, cars, vehicles, machines, tools, household
objects, toys, products, people, or unrelated props as a visual subject.
Never replace an animal with an object that resembles its name or with a
metaphorical illustration. For example, a bee topic must show real bees,
never honey jars, food, soup, or unrelated objects.

Do not use object-only scenes, generic nature scenery, landscapes, flowers,
plants, or empty habitats. A habitat may appear only as background while
the correct animal remains clearly visible. No diagrams or symbolic
substitutions. No space, planets, aliens, vampires, horror, fantasy, or
unrelated footage.

Use real, recognizable animal footage or photorealistic depictions of the
correct species. Keep species, anatomy, colors, and behavior accurate.
Prefer a different animal-focused visual every 1–3 seconds when possible.
Every search_query must name the exact animal and include animal/wildlife
context. Never include food, object, vehicle, or prop terms in search_query.""",
        f'''Create a shot plan for:
TOPIC: {topic}
SCRIPT: {script}

The visuals MUST show only the exact animal species discussed in the
narration. The animal must be visible in every shot and be the main subject.
Do not show food, soup, cooking, dishes, ingredients, bicycles, vehicles,
tools, products, people, objects, or unrelated scenery in any shot.
Do not substitute objects or metaphors for animals. If the narration
mentions an animal behavior involving food, show the animal performing
the behavior, not the food itself.

Reject any visual idea that does not clearly show the correct animal.
Write each search_query as a direct search for real footage of that exact
animal, with no unrelated keywords.

Return exactly:
{{"shots":[{{"sentence":"...","visual":"...","search_query":"...","animation":"...","on_screen_text":"..."}}]}}.''',
        max_output_tokens=3000,
    )
