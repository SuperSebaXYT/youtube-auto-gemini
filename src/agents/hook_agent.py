from src.agents.orchestrator import run_json_batch


def generate_hooks(topic: str, script: str, count: int = 8) -> list[dict]:
    result = run_json_batch(
        "Hook Specialist",
        "Create immediate, accurate, non-clickbait openings for animal and wildlife Shorts.",
        f"""Generate {count} different hooks for this exact animal topic.

TOPIC: {topic}
SCRIPT: {script}

RULES:
- The first sentence must state the most interesting accurate animal fact
  or the central question immediately. No slow scene-setting.
- Make the animal or species clear when needed; do not use a vague "this"
  when the viewer cannot yet know what it refers to.
- A hook may be a bold factual claim or a direct question, but it must not
  exaggerate, invent danger, imply a false ability or promise an absent reveal.
- Do not change species, anatomy, behavior, measurements or evolutionary
  relationships from the supplied script.
- Keep each hook concise, natural when spoken aloud and understandable
  without prior context.
- Make the opening specific to this topic, not a generic curiosity phrase.
- Do not use fake urgency, "scientists are terrified", or unsupported records.

Return {{ "hooks": [{{ "hook": "...", "clarity": 1, "curiosity": 1,
"retention": 1, "science_safety": 1 }}] }}.
"""
        ,
        max_output_tokens=1800,
    )
    return result.get("hooks", []) if isinstance(result, dict) else []
