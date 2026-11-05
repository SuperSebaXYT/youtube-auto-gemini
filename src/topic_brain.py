
"""
topic_brain.py

Cosmic Curious V2 topic discovery engine.

Pipeline:
    candidate generation -> duplicate filtering -> scoring -> winner

Designed to fail safely so the existing pipeline can continue working.

V2 improvements:
- Stronger novelty / overused-topic penalty
- Explicit anti-clickbait and anti-fake-science rules
- Rewards topics with a clear scientific reveal
- Rewards strong visual potential
- Rewards "one surprising question -> one satisfying answer"
- Penalizes topics that are merely broad facts
- Strong scientific accuracy gate
- Penalizes outdated or superseded scientific claims
- Penalizes topics whose hook depends on a misleading premise

Gemini calls are routed through the centralized gemini_client.py
so the project consistently uses the current Chat.send_message()
flow instead of direct client.models.generate_content().
"""

import json
import os
import re
from pathlib import Path

from src.gemini_client import generate, is_available

GEMINI_MODEL = "gemini-3.5-flash-lite"

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"


def _gemini_available():
    """Return True when the configured text provider has an API key."""
    return is_available()


def _clean_lines(text):
    lines = []

    for line in text.splitlines():
        line = re.sub(r"^[\d\.\-\)\s]+", "", line).strip()

        if line:
            lines.append(line)

    return lines


def generate_candidates(config, existing_topics, count=30):
    """
    Generate a large batch of high-quality potential topics.
    """

    if not _gemini_available():
        raise RuntimeError("Configured AI API key is not set")

    existing = "\n".join(
        f"- {topic}"
        for topic in existing_topics[-150:]
    )

    prompt = f"""
You are the head topic researcher for a high-retention faceless YouTube
channel called "{config['display_name']}".

NICHE:
{config['niche']}

CHANNEL STYLE:
{config['tone']}

MISSION:

Generate {count} ORIGINAL video topics that could realistically become
high-performing science/space Shorts.

We are NOT looking for ordinary "science facts".

Each topic should create this reaction:

"Wait... seriously? I need to know why."

CORE REQUIREMENTS:

1. Strong curiosity gap
2. One clear scientific question or phenomenon
3. A surprising but scientifically defensible answer
4. Strong visual storytelling potential
5. Understandable to a normal viewer
6. Enough substance for a 30-60 second Short
7. Potential to expand into a long-form video
8. Prefer topics that viewers probably have NOT already seen repeatedly
9. Prefer specific phenomena over broad categories
10. The answer should be more interesting than the wording of the question

HIGH-VALUE TOPIC PATTERNS:

- Something strange that actually happens in space
- A physical effect with an unintuitive consequence
- An extreme cosmic environment
- A surprising property of a planet, star, moon, black hole, or neutron star
- A real scientific discovery that sounds impossible
- A "what would happen if..." scenario where physics gives a genuinely
  surprising answer
- Something visible or explainable through compelling animation/stock footage
- A familiar concept behaving strangely under extreme conditions

PRIORITIZE:

astronomy
black holes
neutron stars
strange planets
stellar evolution
extreme physics
time
gravity
relativity
space exploration
cosmology
unusual planetary behavior
unknown cosmic objects
weird cosmic environments
real scientific discoveries

IMPORTANT SCIENTIFIC ACCURACY RULE:

The topic itself must remain scientifically defensible even before the
script is written.

Do NOT create a topic whose entire hook depends on an exaggerated,
outdated, disputed, misunderstood, or scientifically misleading claim.

If a famous science claim has been revised by later observations, prefer
the modern understanding rather than the older sensational version.

For example, do NOT build a topic around an extreme dark-matter claim if
later measurements substantially changed that interpretation.

Prefer:

"Why this galaxy once looked like it was almost entirely dark matter"

over:

"The galaxy that is almost entirely dark matter"

when the latter presents an old or disputed measurement as established fact.

IMPORTANT EDITORIAL RULE:

DO NOT choose a topic merely because it sounds dramatic.

Scientific accuracy comes first.

If a claim would require unsupported speculation, reject it.

ANTI-CLICKBAIT:

Avoid titles that promise something the science cannot actually deliver.

Do NOT use fake mystery wording such as:

"Scientists are terrified..."
"This changes everything..."
"What NASA doesn't want you to know..."
"Scientists can't explain..."
"The universe is hiding..."
"Nobody knows..."
"Shocking discovery..."

unless that statement is literally scientifically defensible.

AVOID:

generic "what is X" topics
basic school-level facts
overused black-hole explanations
overused planet facts
listicles
"10 facts about..."
celebrity science
generic NASA news
obvious clickbait
fake science
unsupported speculation
topics with no clear answer
topics that need a huge amount of context before becoming interesting
topics that are primarily philosophical rather than scientific
topics whose interesting claim depends on outdated measurements
topics whose title makes a disputed scientific interpretation sound certain
topics where the premise is technically possible but extremely misleading

BAD:

What is a black hole?
What is gravity?
10 facts about space
Interesting facts about planets
What is the solar system?
What is a neutron star?

MISLEADING:

The Galaxy Made Almost Entirely of Dark Matter
The Planet That Shouldn't Exist
Scientists Have No Idea What This Object Is
A Black Hole Will Destroy Earth Tomorrow

BETTER:

Why a galaxy once appeared to be almost entirely dark matter
Why some planets can survive in places where they shouldn't
Why this strange cosmic object fooled astronomers
What would actually happen if a black hole passed near Earth

EDITORIAL QUALITY TEST:

Before outputting a topic, mentally ask:

"Would someone who already watches science Shorts stop scrolling for this?"

Then ask:

"Can I explain the answer using established science without needing
major caveats that destroy the hook?"

If the answer to either question is no, reject the topic.

ALREADY COVERED:

{existing}

GOOD STYLE EXAMPLES:

Why the sky is dark at night even though there are trillions of stars
What would happen if you fell into a black hole
Why Venus spins backwards compared with most planets
The star that's so big it would swallow Saturn's orbit
Why Mercury has ice despite temperatures hot enough to melt lead

Generate exactly {count} ideas.

Each must be:

- ORIGINAL
- specific
- scientifically defensible
- visually explainable
- curiosity-driven
- based on modern scientific understanding

Return ONLY one topic per line.

No numbering.
No explanations.
No markdown.
No quotes.
"""

    text = generate(
        prompt,
        model=GEMINI_MODEL,
        max_output_tokens=2000,
    )

    return _clean_lines(text)


def score_candidates(config, candidates):
    """
    Score candidates using a stronger editorial model.

    Scores:
        Novelty
        Curiosity
        Science
        Visual
        Shorts
        Longform
        Accuracy
        Specificity

    The final score remains normalized to 100.
    """

    if not _gemini_available():
        raise RuntimeError("GEMINI_API_KEY is not set")

    topics = "\n".join(
        f"{i+1}. {topic}"
        for i, topic in enumerate(candidates)
    )

    prompt = f"""
You are the senior editorial director of "{config['display_name']}".

Your job is to select the ONE science/space topic most likely to produce
a high-retention YouTube Short WITHOUT sacrificing scientific accuracy.

Evaluate every candidate carefully.

Score each dimension from 1-10:

NOVELTY
How uncommon and fresh is the idea?

CURIOSITY
How strongly does the title make someone need the answer?

SCIENCE
How scientifically meaningful and interesting is the phenomenon?

VISUAL
Can this be shown using compelling space imagery, animation, diagrams,
or stock footage?

SHORTS
Can the concept be explained clearly and powerfully in 30-60 seconds?

LONGFORM
Could the topic later become a strong 5-10 minute video?

ACCURACY
How safely can the topic be explained without misleading viewers?

SPECIFICITY
Does the topic describe one concrete phenomenon rather than a broad
educational category?

WEIGHTING:

Novelty       18%
Curiosity     22%
Science       14%
Visual        14%
Shorts        12%
Longform       7%
Accuracy       8%
Specificity    5%

TOTAL must be calculated from these weights and normalized to 100.

==================================================
CRITICAL SCIENTIFIC ACCURACY RULE
==================================================

A topic MUST NOT receive a high score simply because its wording is
dramatic.

Before scoring the other dimensions, mentally fact-check the premise.

Ask:

1. Is the central claim consistent with modern scientific understanding?
2. Is it based on established evidence rather than speculation?
3. Could the topic title cause an ordinary viewer to believe something
   scientifically false?
4. Is the claim outdated, superseded, disputed, or dependent on an
   early measurement that was later revised?
5. Would the final script need so many caveats that the original hook
   becomes misleading?

If the answer raises serious concerns, reduce ACCURACY substantially.

A topic with an exciting hook but a scientifically misleading premise
should lose to a slightly less dramatic topic with a rock-solid premise.

==================================================
NO OUTDATED SCIENCE
==================================================

Do NOT reward topics built around famous scientific claims that have
since been substantially revised.

For example:

If early observations suggested an object had an extreme property but
later observations showed that the original interpretation was wrong,
do NOT score the sensational interpretation as if it were established
fact.

Instead, a revised version can be valuable if the real story is:

"Why astronomers once thought X — and what better measurements revealed."

This kind of topic can actually receive a NOVELTY bonus because it
contains a surprising scientific reversal.

==================================================
NO MISLEADING PREMISES
==================================================

Penalize topics where:

- the title exaggerates the actual phenomenon
- a possibility is presented as certainty
- a theoretical scenario is presented as an observed fact
- an outdated measurement is treated as current
- a disputed interpretation is treated as settled
- the dramatic wording is stronger than the underlying science
- the viewer would likely leave with the wrong scientific understanding

IMPORTANT:

Do NOT reject a topic merely because it involves a hypothetical
scenario.

Hypothetical topics are allowed when the scenario follows established
physics and the answer can be explained accurately.

For example:

"What would happen if a rogue planet entered our Solar System?"

can be excellent.

But:

"A rogue planet WILL destroy Earth"

should be heavily penalized unless there is actual evidence supporting
that claim.

==================================================
CURIOSITY VS ACCURACY
==================================================

Curiosity is valuable, but accuracy is a hard constraint.

Never compensate for poor scientific accuracy with high curiosity.

A topic scoring:

Curiosity = 10
Accuracy = 4

should generally lose to a topic scoring:

Curiosity = 8
Accuracy = 9

when the first topic's hook depends on misleading science.

==================================================
ANTI-CLICKBAIT
==================================================

Do NOT reward:

"Scientists are terrified..."
"This changes everything..."
"NASA doesn't want you to know..."
"Scientists can't explain..."
"Nobody knows..."
"The universe is hiding..."
"Shocking discovery..."

unless the statement is literally and defensibly true.

==================================================
BONUS FOR SCIENTIFIC REVERSALS
==================================================

Give a NOVELTY and CURIOSITY bonus to topics where:

- scientists initially believed one thing
- better observations revealed something different
- the correction itself is scientifically interesting

These topics are especially valuable because they provide a natural
story structure:

INITIAL BELIEF -> SURPRISE -> BETTER EVIDENCE -> NEW UNDERSTANDING

==================================================
BONUS FOR ONE CLEAN REVEAL
==================================================

Favor topics that can be structured as:

HOOK
    ↓
QUESTION
    ↓
BUILD CURIOSITY
    ↓
SCIENTIFIC REVEAL
    ↓
SURPRISING CONSEQUENCE

Avoid topics that require five unrelated concepts before the viewer
understands the interesting part.

==================================================
VISUAL STORY RULE
==================================================

Prefer topics where the visuals can meaningfully evolve every few
seconds.

Strong examples include:

- approaching a black hole
- a planet changing appearance
- a star expanding or collapsing
- gravitational effects
- orbital changes
- extreme planetary environments
- cosmic collisions
- scale comparisons
- simulations of physical processes

Do not give a high VISUAL score merely because "space footage" can be
used in the background.

The phenomenon itself should provide opportunities for visual storytelling.

==================================================
PENALTIES
==================================================

PENALIZE:

- generic educational questions
- topics already heavily covered online
- topics where the answer is obvious
- vague "space is amazing" concepts
- topics dependent on speculation
- misleading claims
- impossible scenarios presented as established science
- outdated scientific claims
- disputed claims presented as fact
- topics with weak visual possibilities
- topics that require excessive setup
- topics where the title is more interesting than the actual science

==================================================
FINAL EDITORIAL TEST
==================================================

Before assigning the final score, ask:

"Would a science-interested viewer stop scrolling?"

Then:

"Would a scientifically knowledgeable viewer consider the premise
fair and accurate?"

Then:

"Can the entire idea be delivered in roughly 30-60 seconds without
misleading the audience?"

Only topics that pass all three should be considered strong winners.

Return ONLY valid JSON.

Exact structure:

[
  {{
    "topic": "...",
    "novelty": 0,
    "curiosity": 0,
    "science": 0,
    "visual": 0,
    "shorts": 0,
    "longform": 0,
    "accuracy": 0,
    "specificity": 0,
    "total": 0,
    "reason": "one short sentence"
  }}
]

TOPICS:

{topics}
"""

    text = generate(
        prompt,
        model=GEMINI_MODEL,
        max_output_tokens=3000,
    )

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    return json.loads(text)


def choose_winner(config, candidates):
    """
    Score all candidates and select the highest-quality topic.
    """

    if not candidates:
        return None

    scored = score_candidates(config, candidates)

    if not scored:
        return candidates[0]

    scored.sort(
        key=lambda item: item.get("total", 0),
        reverse=True,
    )

    winner = scored[0]

    print("\n=== TOPIC BRAIN ===")

    for item in scored[:5]:
        print(
            f"{item.get('total', 0):>5}/100  "
            f"{item.get('topic', '')}"
        )

    print(
        f"\nWINNER: {winner['topic']}"
        f" ({winner.get('total', 0)}/100)"
    )

    reason = winner.get("reason")

    if reason:
        print(f"Reason: {reason}")

    return winner["topic"]


def discover_topic(config, existing_topics):
    """
    Main safe entry point.

    Returns:
        topic string
        or None if Topic Brain fails.
    """

    try:
        candidates = generate_candidates(
            config,
            existing_topics,
            count=30,
        )

        if not candidates:
            return None

        # Remove exact duplicates.
        candidates = list(dict.fromkeys(candidates))

        print(
            f"Topic Brain: generated {len(candidates)} candidates"
        )

        return choose_winner(
            config,
            candidates,
        )

    except Exception as e:
        print(
            f"WARNING: Topic Brain failed ({e}); "
            f"falling back to existing topic system."
        )

        return None
