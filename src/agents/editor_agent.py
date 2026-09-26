"""Final opportunity-ranking agent for Animals & Wildlife."""


from src.agents.orchestrator import run_json_batch


def select_winners(
    finalists: list[dict],
    config: dict,
    count: int = 5,
    performance_summary: str = None,
) -> dict:

    packed = "\n\n".join(
        f"INDEX: {i + 1}\n"
        f"TOPIC: {x['topic']}\n"
        f"CURIOSITY: {x.get('curiosity', 0)}\n"
        f"NOVELTY: {x.get('novelty_score', x.get('novelty', 0))}\n"
        f"VISUAL: {x.get('visual', 0)}\n"
        f"FACT CONFIDENCE: "
        f"{x.get('science_confidence', x.get('accuracy_score', 0))}\n"
        f"ORIGINALITY NOTE: {x.get('novelty_reason', '')}\n"
        f"FACT-CHECK NOTE: {x.get('safe_framing', '')}"
        for i, x in enumerate(finalists)
    )

    performance_block = ""

    if performance_summary:
        performance_block = f"""

REAL CHANNEL PERFORMANCE DATA (use this to inform your ranking -- prefer
candidates that resemble the style, subject, or angle of what has genuinely
performed well on THIS channel before, and be more cautious about patterns
resembling what has underperformed):

{performance_summary}
"""

    return run_json_batch(
        "Chief Editor",
        "Choose the strongest animal and wildlife video opportunities while protecting long-term channel quality.",
        f"""
Rank the candidates and choose the top {count}.

IMPORTANT:
ALL selected topics MUST be about animals or wildlife.

Do NOT select topics about:
- Space
- Planets
- Stars
- Galaxies
- Black holes
- Astronomy
- NASA
- Physics unrelated to animals
- Generic science unrelated to animals

Use these weighted dimensions:

- curiosity 35%
- novelty 25%
- visual potential 20%
- factual confidence 20%

{performance_block}

SCORING RULES:
- "score" must be an integer from 1 to 100.
- Scores MUST reflect genuine relative differences between candidates.
- Do NOT assign identical or near-identical scores to every candidate.
- A ranked list should show meaningful differences in strength.
- "rank" 1 must have the highest score.
- "rank" {count} must have the lowest score among the selected picks.

The selected topics should be:
- surprising
- visually interesting
- based on real animal or wildlife facts
- suitable for YouTube Shorts
- easy to understand
- engaging without misleading viewers

Return JSON object:

{{
  "winners": [
    {{
      "topic": "...",
      "score": 0,
      "rank": 1,
      "why": "...",
      "recommended_format": "short" or "both"
    }}
  ],
  "winner": "best topic",
  "strategy_note": "one short note"
}}

CANDIDATES:
{packed}
""",
        f"""
CHANNEL: {config['display_name']}
NICHE: Animals & wildlife facts
TONE: {config['tone']}
""",
        max_output_tokens=3500,
    )