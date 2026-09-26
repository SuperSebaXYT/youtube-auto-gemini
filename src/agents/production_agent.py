"""Production brief agent for Animals & Wildlife.

Turns one approved animal or wildlife topic into a structured handoff
for the existing video-production pipeline. This is planning only;
it does not upload.
"""

from src.agents.orchestrator import run_json_batch


def build_brief(
    topic: str,
    winner: dict,
    config: dict,
) -> dict:

    return run_json_batch(
        "Production Director",
        "Turn an approved animal or wildlife topic into a production-ready creative brief without inventing facts.",
        f"""
Create a production brief for this approved animal/wildlife topic.

IMPORTANT:
The content must stay focused on animals and wildlife.

TOPIC:
{topic}

EDITOR WINNER DATA:
{winner}

Return exactly this JSON shape:

{{
  "topic": "...",
  "format": "short" or "both",
  "core_question": "...",
  "animal_reveal": "...",
  "safe_framing": "...",
  "hook_options": ["...", "...", "...", "...", "..."],
  "preferred_hook": "...",
  "narration_beats": [
    {{"beat": 1, "purpose": "hook", "seconds": 3, "content": "..."}},
    {{"beat": 2, "purpose": "setup", "seconds": 5, "content": "..."}},
    {{"beat": 3, "purpose": "mechanism", "seconds": 10, "content": "..."}},
    {{"beat": 4, "purpose": "reveal", "seconds": 10, "content": "..."}},
    {{"beat": 5, "purpose": "ending", "seconds": 5, "content": "..."}}
  ],
  "visual_beats": [
    {{"beat": 1, "visual": "...", "stock_query": "...", "ai_visual": false}},
    {{"beat": 2, "visual": "...", "stock_query": "...", "ai_visual": false}},
    {{"beat": 3, "visual": "...", "stock_query": "...", "ai_visual": true}}
  ],
  "thumbnail_concepts": [
    {{"concept": "...", "text": "...", "focal_object": "..."}},
    {{"concept": "...", "text": "...", "focal_object": "..."}},
    {{"concept": "...", "text": "...", "focal_object": "..."}}
  ],
  "title_options": ["...", "...", "...", "...", "..."],
  "description_angle": "...",
  "fact_risks": ["..."],
  "success_hypothesis": "..."
}}

RULES:

- The topic MUST be about animals or wildlife.
- Do not introduce space, astronomy, planets, stars, galaxies,
  black holes, NASA, or unrelated science.
- Keep factual claims defensible.
- Do not invent observations, numbers, discoveries, or citations.
- Do not exaggerate animal abilities or behavior.
- Use conditional language when something is uncertain.
- Prefer specific animal visuals over generic nature footage.
- Make stock queries specific enough to find the correct animal.
- Keep the Short plan compatible with 30-60 seconds.
- Make the hook surprising and conversational.
- The final result should feel like an amazing fact being told to a friend,
  not like a school textbook.

""",
        f"""
CHANNEL: {config['display_name']}
NICHE: Animals & wildlife facts
TONE: {config['tone']}
""",
        max_output_tokens=4500,
    )