"""Batch factual screening for animal and wildlife topic candidates."""

from src.agents.orchestrator import run_json_batch


def fact_check_topics(topics: list[str]) -> list[dict]:
    candidate_text = "\n".join(
        f"{i + 1}. {x}"
        for i, x in enumerate(topics)
    )

    result = run_json_batch(
        "Animal Fact Guardian",
        "Fact-check animal and wildlife topic premises before production; factual accuracy is a hard constraint.",
        f"""
Evaluate ALL topics in one batch.

IMPORTANT:
These topics are for an animal and wildlife facts channel.

Check whether each topic is:
- factually supported
- about animals or wildlife
- not misleading or exaggerated
- suitable for a general YouTube audience

Reject topics that:
- contain false animal facts
- make exaggerated claims presented as facts
- rely on unsupported mysteries
- are about space, astronomy, planets, stars, galaxies, NASA,
  or unrelated science
- are not meaningfully about animals or wildlife

Do not reject ordinary simplification for a general audience
when the simplified statement remains accurate.

Return JSON object:

{{
  "results": [
    {{
      "index": 1,
      "verdict": "PASS" or "REVISE" or "REJECT",
      "accuracy_score": 1,
      "claim_type": "established" or "hypothesis" or "mixed",
      "key_risk": "short description or empty",
      "safe_framing": "recommended wording"
    }}
  ]
}}

TOPICS:
{candidate_text}
""",
        max_output_tokens=3500,
    )

    return (
        result.get("results", [])
        if isinstance(result, dict)
        else []
    )