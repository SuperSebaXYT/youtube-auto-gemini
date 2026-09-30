from src.agents.orchestrator import run_json_batch


def plan(topic: str, script: str) -> dict:
    return run_json_batch(
        "Visual Director",
        """Create scene-specific visuals that match the narration exactly.
The channel is about animals and wildlife facts.
Never suggest space, planets, aliens, vampires, horror, fantasy, or unrelated footage.
Use real animals, wildlife, nature, habitats, animal behavior, and close-up shots
that directly match what is being said.
Prefer a different visual every 1–3 seconds when possible.
Search queries should be concrete and suitable for finding real animal footage.""",
        f'''Create a shot plan for:
TOPIC: {topic}
SCRIPT: {script}

The visuals MUST match the animal being discussed and the exact narration.
Do not introduce unrelated animals or fictional subjects.

Return exactly:
{{"shots":[{{"sentence":"...","visual":"...","search_query":"...","animation":"...","on_screen_text":"..."}}]}}.''',
        max_output_tokens=3000,
    )