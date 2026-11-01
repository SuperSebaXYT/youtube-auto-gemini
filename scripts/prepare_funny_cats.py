from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch_file(relative_path: str, replacements: list[tuple[str, str, int | None]]) -> None:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    for old, new, expected in replacements:
        count = text.count(old)
        if count == 0 or (expected is not None and count != expected):
            raise RuntimeError(
                f"Patch mismatch in {relative_path}: expected "
                f"{expected if expected is not None else 'at least 1'} "
                f"match(es), found {count} for {old[:100]!r}"
            )
        text = text.replace(old, new, expected if expected is not None else -1)
    path.write_text(text, encoding="utf-8")
    print(f"Prepared Funny Cats engine: {relative_path}")


patch_file("src/generate_script.py", [
    (
        "NICHE: Animals & wildlife facts",
        "NICHE: {config.get('niche', 'Animals & wildlife facts')}",
        1,
    ),
    (
        "Every topic must concern real animals or wildlife. Explore different\nspecies, senses, behaviors, survival adaptations, intelligence, predators,\nprey, rare animals, deep-sea animals, and unusual animal relationships.\n\nDo not generate space, astronomy, NASA, unrelated science, or fictional\nanimal abilities. Avoid duplicate subjects and close rewordings.",
        "{config.get('topic_generation_rules', 'Generate fresh topics within the configured channel niche. Avoid duplicates and close rewordings.')}",
        1,
    ),
    (
        "if not topic:\n        try:\n            candidate = _generate_winner_inspired_topic(",
        "if not topic and config.get('use_winner_hunter', True):\n        try:\n            candidate = _generate_winner_inspired_topic(",
        1,
    ),
    (
        "NICHE: Animals and wildlife facts.\nTONE: Curious, surprising, energetic, punchy, conversational.",
        "NICHE: {config.get('niche', 'Animals and wildlife facts')}\nTONE: {config.get('tone', 'curious, surprising, energetic, punchy, conversational')}",
        1,
    ),
    (
        "CONTENT RULES:\n- The entire script must be about animals or wildlife.\n- Use real, well-established facts.\n- Never invent statistics, discoveries, quotes, experiments, behaviors,\n  or abilities.\n- Never exaggerate abilities or present speculation as fact.\n- Do not write about space, planets, stars, galaxies, black holes,\n  astronomy, NASA, the universe, or unrelated physics.\n- Stay focused on the specific animal and topic.\n- Do not invent a \"shocking\" fact just to make the script exciting.",
        "CONTENT RULES:\n{config.get('script_rules', 'Stay within the configured channel niche. Be accurate, engaging, and avoid unsupported claims.')}",
        1,
    ),
    (
        "- Start with a concrete, surprising fact or consequence about the animal.",
        "- Start with a funny, instantly relatable cat moment or an unexpected feline behavior.",
        1,
    ),
    (
        "- Every sentence must introduce a new fact, escalation, contrast, or payoff.",
        "- Every sentence must add a new joke, escalation, relatable detail, or payoff.",
        1,
    ),
    (
        "- Build one clear curiosity loop and resolve it before the ending.",
        "- Build one clear comedic setup and deliver the payoff before the ending.",
        1,
    ),
    (
        "- The payoff must be specific and supported by a real fact.",
        "- The payoff must be a clear comedic twist or relatable cat behavior.",
        1,
    ),
])

patch_file("src/fetch_stock.py", [
    (
        '        "animal",\n        "animals",',
        '        "animal",\n        "animals",\n        "cat",\n        "cats",\n        "kitten",\n        "kittens",\n        "feline",\n        "felines",',
        2,
    ),
    (
        "    queries = []\n\n    # ---------------------------------------------------------\n    # SPECIAL CASE: SEA CUCUMBER",
        '    queries = []\n    cat_topic = bool(set(words).intersection({"cat", "cats", "kitten", "kittens", "feline", "felines"}))\n\n    if cat_topic:\n        queries.extend([\n            "funny cats",\n            "cats being funny",\n            "cats playing",\n            "funny kittens",\n            "cat reactions",\n            "cats at home",\n        ])\n\n    # ---------------------------------------------------------\n    # SPECIAL CASE: SEA CUCUMBER',
        1,
    ),
    (
        "    if (\n        any(\n            word in animal_words",
        "    if (\n        not cat_topic\n        and any(\n            word in animal_words",
        1,
    ),
    (
        '    queries.extend([\n        "nature wildlife",\n        "animals nature",\n        "documentary nature",\n        "cinematic nature",\n        "nature close up",\n    ])',
        '    if not cat_topic:\n        queries.extend([\n            "nature wildlife",\n            "animals nature",\n            "documentary nature",\n            "cinematic nature",\n            "nature close up",\n        ])',
        1,
    ),
])

patch_file("src/main.py", [
    (
        'cta = "Subscribe for more amazing animal facts every day! 🐾"',
        'cta = "Subscribe for more funny cat videos every day! 🐾"',
        1,
    ),
    (
        "# QUALITY GATE\n        # -----------------------------------------------------\n\n        if not is_kids_channel:",
        "# QUALITY GATE\n        # -----------------------------------------------------\n\n        if not is_kids_channel and config.get('use_quality_gate', True):",
        1,
    ),
])
