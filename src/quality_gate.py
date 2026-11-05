
"""
quality_gate.py

Final scientific/editorial quality gate for generated YouTube scripts.

Pipeline:
    generated script
        ↓
    deterministic checks
        ↓
    scientific fact-check
        ↓
    accuracy / certainty / hypothesis audit
        ↓
    approve OR repair
        ↓
    re-check repaired script

Designed to fail safely:
- Bad scripts are repaired before TTS/video assembly.
- If the repaired script still fails, publishing is blocked.
- Scientific accuracy takes priority over sensationalism.

V3 improvements:
- Stronger claim-by-claim scientific audit
- Absolute-language detection is advisory rather than an automatic failure
- Hypothetical scenarios are checked for conditional wording
- Numerical claims receive special scrutiny
- Cause/effect claims are checked
- Outdated terminology and scientific interpretations are checked
- Misleading hooks are checked
- Repair instructions explicitly address each failed claim
- Uses Chat.send_message() for Gemini text generation
"""

import json
import os
import re
import time

from google import genai as genai_client


# Keep this synchronized with the rest of the project.
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Initial check + 2 repairs = maximum 3 checks.
MAX_REPAIR_ATTEMPTS = 2

# Minimum score required for approval.
MIN_APPROVAL_SCORE = 8.0


def _send_message_with_retry(chat, prompt: str, max_attempts: int = 4):
    """Retry transient Gemini capacity, rate-limit, and server errors."""
    retryable_markers = (
        "429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
        "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
    )
    for attempt in range(max_attempts):
        try:
            return chat.send_message(prompt)
        except Exception as exc:
            message = str(exc).upper()
            retryable = any(marker in message for marker in retryable_markers)
            if not retryable or attempt == max_attempts - 1:
                raise
            delay = min(30, 3 * (2 ** attempt))
            print(
                f"WARNING: Gemini transient error; retrying in {delay}s "
                f"(attempt {attempt + 2}/{max_attempts}): {exc}"
            )
            time.sleep(delay)


def _client():
    """
    Create a Gemini client.
    """

    key = os.environ.get("GEMINI_API_KEY")

    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    return genai_client.Client(api_key=key)


def _chat(client):
    """
    Create a Gemini chat session.

    IMPORTANT:
    Use Chat.send_message() rather than client.models.generate_content()
    for normal text generation.
    """

    return client.chats.create(
        model=GEMINI_MODEL
    )


def _clean_json(text: str) -> dict:
    """
    Extract JSON even if Gemini accidentally wraps it in markdown fences.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove opening markdown fence.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing markdown fence.
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            f"Gemini did not return JSON: {text[:500]}"
        )

    try:
        return json.loads(
            text[start:end + 1]
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON: {e}. "
            f"Response: {text[:500]}"
        ) from e


def _word_count(text: str) -> int:
    """
    Count spoken words approximately.
    """

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
        )
    )


def _basic_script_checks(script: str, target_words: int = 150) -> list:
    """
    Cheap deterministic checks before spending another Gemini call.

    IMPORTANT:
    This function intentionally does NOT automatically reject normal
    scientific words such as:

        will
        never
        always
        instantly
        completely

    Those words can be scientifically correct in the right context.

    Gemini performs the actual scientific certainty audit.

    Deterministic checks are reserved for obvious formatting/template
    problems.
    """

    problems = []

    if not script or len(script.strip()) < 80:
        problems.append(
            "Script is too short."
        )

    words = _word_count(script)

    # Tolerance band around the actual target length instead of a
    # hardcoded Short-sized 70-150 range -- this was silently rejecting
    # every genuine long-form script (e.g. a 371-word 5-minute script)
    # by judging it against Short-length bounds it was never supposed
    # to meet, regardless of what target_words was actually passed in.
    min_words = round(target_words * 0.8)
    max_words = round(target_words * 1.5)

    if words < min_words:
        problems.append(
            f"Only {words} words; too short for the target of "
            f"~{target_words} words."
        )

    if words > max_words:
        problems.append(
            f"{words} words; too long for the target of "
            f"~{target_words} words."
        )

    # ------------------------------------------------------------
    # Detect accidental word joining.
    #
    # These usually happen during generation or cleanup when spaces
    # disappear between words.
    # ------------------------------------------------------------

    joined_words = [
        "reasonsbehind",
        "flashof",
        "deepspace",
        "darknessof",
        "becauseof",
        "partof",
        "typeof",
        "kindof",
        "endof",
        "highresolution",
        "lightyearsaway",
        "millionyear",
        "millionyears",
        "billionyear",
        "billionyears",
        "spacebetween",
        "closeto",
        "farfrom",
        "aroundthe",
        "insideof",
        "outsideof",
        "backto",
        "morethan",
        "lessthan",
    ]

    lower = script.lower()

    for word in joined_words:
        if word in lower:
            problems.append(
                f"Accidental joined word detected: {word}"
            )

    # ------------------------------------------------------------
    # Repeated punctuation.
    # ------------------------------------------------------------

    if "??" in script or "!!" in script:
        problems.append(
            "Repeated punctuation detected."
        )

    # ------------------------------------------------------------
    # Generic/template phrases.
    #
    # These are deterministic because they are clearly undesirable
    # regardless of scientific context.
    # ------------------------------------------------------------

    bad_phrases = [
        "here's something most people don't know",
        "here is something most people don't know",
        "it sounds simple, but",
        "you'll start noticing it everywhere",
        "that's the kind of small insight",
        "in conclusion",
        "subscribe for more",
        "like and subscribe",
        "don't forget to subscribe",
        "smash that like button",
        "follow for more",
    ]

    for phrase in bad_phrases:
        if phrase in lower:
            problems.append(
                f"Generic/template phrase detected: {phrase}"
            )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # We deliberately DO NOT reject words like:
    #
    # always
    # never
    # instantly
    # completely
    # will
    # definitely
    #
    # here.
    #
    # A sentence such as:
    #
    # "Light from the Sun will eventually reach Earth."
    #
    # can be perfectly valid.
    #
    # Gemini's claim-by-claim scientific audit below is responsible
    # for determining whether certainty wording is actually wrong.
    # ------------------------------------------------------------

    return problems


def check_script(
    topic: str,
    script: str,
    config: dict,
    target_words: int = 150,
) -> dict:
    """
    Ask Gemini to perform a strict scientific/editorial audit.
    """

    basic_problems = _basic_script_checks(
        script, target_words
    )

    client = _client()
    chat = _chat(client)

    prompt = f"""
You are the FINAL scientific fact-checker and editorial gatekeeper for
a professional faceless YouTube science channel.

CHANNEL:
{config['display_name']}

NICHE:
{config['niche']}

TONE:
{config['tone']}

TOPIC:
{topic}

SCRIPT:
{script}

Your decision determines whether this script is allowed to proceed to
voice generation and publication.

This is NOT primarily a creative-writing evaluation.

SCIENCE ACCURACY IS THE HIGHEST PRIORITY.

==================================================
STEP 1 — CLAIM-BY-CLAIM FACT CHECK
==================================================

Read every sentence and identify factual claims.

For each important claim, ask:

- Is it scientifically correct?
- Is it consistent with modern scientific understanding?
- Is it supported by established physics/astronomy?
- Is the wording stronger than the evidence?
- Does the claim depend on conditions that the script ignores?
- Is the claim outdated or based on an older interpretation?
- Is a possibility being presented as a certainty?

Do NOT simply ask whether the overall story "sounds plausible".

A single scientifically misleading central claim is enough to reject
the script.

==================================================
STEP 2 — HYPOTHETICAL SCENARIOS
==================================================

Hypothetical science is ALLOWED and encouraged when interesting.

For example:

"What would happen if a rogue planet entered our Solar System?"

can be an excellent topic.

But distinguish carefully between:

A) established physical behavior

and

B) one possible outcome under a particular hypothetical configuration.

Do NOT allow a hypothetical outcome to be presented as guaranteed when
the result depends on variables such as:

- mass
- distance
- velocity
- angle
- orbit
- composition
- energy
- luminosity
- timing
- density
- magnetic field
- initial conditions

Good:

"A sufficiently massive companion could disturb the Oort Cloud."

Bad:

"A second star would send comets toward Earth."

If the result is strongly dependent on conditions, the wording should
reflect that with phrases such as:

- could
- might
- under these conditions
- depending on
- in some scenarios
- simulations suggest
- researchers have proposed

Use conditional language only when scientifically appropriate.

Do NOT weaken statements that are genuinely established facts.

==================================================
STEP 3 — ABSOLUTE CLAIM AUDIT
==================================================

Words such as:

always
never
will
guaranteed
definitely
impossible
instantly
completely
every
none
the entire
the only
exactly

are NOT automatically errors.

Evaluate them IN CONTEXT.

For example:

"Nothing can escape from inside an event horizon."

may be scientifically defensible when correctly referring to signals
escaping to the outside universe.

But:

"A black hole always destroys anything nearby instantly."

would be misleading.

Do not reject a script merely because it contains an absolute word.

Reject it only if the actual scientific claim is materially
overstated.

==================================================
STEP 4 — OUTDATED SCIENCE
==================================================

Check whether the script relies on a famous but outdated scientific
claim.

If historical measurements or interpretations were later revised,
the script must NOT present the old interpretation as current
scientific fact.

A scientifically interesting correction is encouraged.

==================================================
STEP 5 — NUMBERS AND SCALE
==================================================

Pay special attention to:

- distances
- masses
- speeds
- temperatures
- time scales
- percentages
- frequencies
- energy
- number of objects
- comparisons with Earth
- comparisons with the Sun
- comparisons with the Milky Way

If a numerical statement looks suspicious, flag it.

Do not invent a correction.

If exact verification is unavailable, recommend safer wording.

A simple approximate statement is preferable to a fabricated precise
number.

==================================================
STEP 6 — CAUSE AND EFFECT
==================================================

Do not allow the script to turn correlation or possibility into
guaranteed cause and effect.

Look for statements such as:

"X causes Y"

"X would definitely cause Y"

"X means Y"

"X proves Y"

when the actual science is more conditional.

==================================================
STEP 7 — SCIENTIFIC TERMINOLOGY
==================================================

Flag terminology that is outdated, technically incorrect, or commonly
misunderstood.

Examples include:

- treating relativistic mass as ordinary modern mass
- confusing gravity with gravitational waves
- confusing mass with weight
- confusing an event horizon with a physical surface
- confusing possibility with probability
- treating theoretical predictions as observations
- using "zero gravity" when microgravity/free fall is more accurate

==================================================
STEP 8 — MISLEADING HOOK TEST
==================================================

The first sentence is extremely important.

Ask:

"If a viewer watches the whole Short, will they understand what the
hook actually meant?"

Reject hooks that deliberately create a scientifically false
impression.

The hook may be dramatic.

It may NOT be scientifically deceptive.

==================================================
STEP 9 — VISUAL CLAIM TEST
==================================================

The script will be paired with stock footage and generated visuals.

The scientific phenomenon itself should be visually explainable.

Do not reject a scientifically correct script merely because stock
footage might be imperfect.

==================================================
STEP 10 — SHORT-FORM QUALITY
==================================================

The script should:

- hook immediately
- introduce one clear question
- build curiosity
- explain one central scientific idea
- deliver a satisfying reveal
- end strongly

Avoid unnecessary background information.

Avoid multiple unrelated scientific concepts.

==================================================
CRITICAL DECISION RULE
==================================================

Accuracy is a HARD CONSTRAINT.

A dramatic script with questionable science must lose to a slightly
less dramatic script with excellent science.

Do NOT compensate for inaccurate science with high curiosity.

A strong script should generally have:

Accuracy >= 8
Science >= 8
Overall >= 8

==================================================
IMPORTANT SCORING RULE
==================================================

Do not automatically reject a script simply because:

- it uses "will"
- it uses "never"
- it uses "always"
- it uses "instantly"
- it uses "completely"
- it contains a hypothetical scenario

Judge the COMPLETE CLAIM.

Only flag such wording when it creates a genuine scientific
overstatement or materially misleading impression.

==================================================
APPROVAL STANDARD
==================================================

APPROVE only if:

1. The central scientific claim is accurate.
2. Important supporting claims are accurate.
3. Hypothetical scenarios are clearly treated as hypothetical when
   their outcomes are conditional.
4. Conditional outcomes use appropriately conditional language.
5. There are no materially misleading statements.
6. There are no important outdated scientific claims presented as
   current.
7. Numerical claims are reasonable.
8. The script sounds natural.
9. The script works as short-form science narration.

REJECT if any major scientific claim is wrong or materially misleading.

==================================================
SCORING
==================================================

Give an overall score from 1-10.

Consider:

- scientific accuracy
- scientific significance
- clarity
- curiosity
- retention potential
- visual storytelling
- narration quality

But NEVER allow entertainment value to hide a major factual problem.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON.

Use exactly:

{{
  "approved": true,
  "score": 9.2,
  "problems": [],
  "corrections": [],
  "reason": "Short explanation"
}}

OR:

{{
  "approved": false,
  "score": 6.2,
  "problems": [
    "Specific factual problem"
  ],
  "corrections": [
    "Specific scientifically accurate correction"
  ],
  "reason": "Short explanation"
}}

IMPORTANT:

Problems must be specific.

Corrections must explain what the writer should change.

Do NOT rewrite the entire script.

Do NOT reject normal certainty words unless they create an actual
scientific error.

Do NOT invent facts in order to justify a rejection.
"""

    response = _send_message_with_retry(
        chat, prompt
    )

    result = _clean_json(
        response.text
    )

    # ------------------------------------------------------------
    # Normalize Gemini output.
    # ------------------------------------------------------------

    if not isinstance(result, dict):
        raise ValueError(
            "Quality Gate response was not a JSON object."
        )

    result.setdefault(
        "problems",
        []
    )

    result.setdefault(
        "corrections",
        []
    )

    result.setdefault(
        "reason",
        ""
    )

    # ------------------------------------------------------------
    # Add deterministic problems.
    #
    # These are hard failures because they are structural or editorial
    # issues that Gemini should not override.
    # ------------------------------------------------------------

    if basic_problems:

        for problem in basic_problems:

            if problem not in result["problems"]:
                result["problems"].append(
                    problem
                )

        result["approved"] = False

    # ------------------------------------------------------------
    # Normalize score.
    # ------------------------------------------------------------

    try:
        result["score"] = float(
            result.get(
                "score",
                0
            )
        )
    except Exception:
        result["score"] = 0.0

    # Clamp score to 0-10.
    result["score"] = max(
        0.0,
        min(
            10.0,
            result["score"]
        )
    )

    # ------------------------------------------------------------
    # Final approval rules.
    # ------------------------------------------------------------

    if result["score"] < MIN_APPROVAL_SCORE:
        result["approved"] = False

    if result.get("problems"):
        result["approved"] = False

    return result


def repair_script(
    topic: str,
    script: str,
    review: dict,
    config: dict,
    target_words: int,
) -> str:
    """
    Rewrite a failed script while preserving the interesting idea.
    """

    client = _client()
    chat = _chat(client)

    problems = "\n".join(
        f"- {p}"
        for p in review.get(
            "problems",
            []
        )
    )

    corrections = "\n".join(
        f"- {c}"
        for c in review.get(
            "corrections",
            []
        )
    )

    # If Gemini returned no corrections, explicitly tell the repair model
    # to independently fix the listed problems without inventing science.
    if not corrections:
        corrections = (
            "- Correct every factual problem identified by the fact-check.\n"
            "- If a claim depends on conditions, use scientifically "
            "appropriate conditional wording."
        )

    prompt = f"""
You are a senior science editor repairing a YouTube Short that failed
scientific fact-checking.

CHANNEL:
{config['display_name']}

NICHE:
{config['niche']}

TOPIC:
{topic}

ORIGINAL SCRIPT:
{script}

FACT-CHECK PROBLEMS:
{problems}

REQUIRED CORRECTIONS:
{corrections}

Your job is to produce a corrected version that is both exciting and
scientifically defensible.

==================================================
NON-NEGOTIABLE SCIENCE RULES
==================================================

1. Never invent facts.

2. Never exaggerate a physical effect.

3. Never turn a possibility into a certainty.

4. Never present a hypothetical scenario as an observed fact.

5. If the outcome depends on variables such as mass, distance, speed,
   orbit, luminosity or initial conditions, use appropriately
   conditional wording.

6. Do not use outdated scientific terminology when a modern term is
   available.

7. Do not present old measurements or superseded interpretations as
   current scientific consensus.

8. If an exact numerical claim is uncertain, remove the number or use
   a safer scientifically defensible comparison.

9. Preserve the interesting idea whenever possible.

10. The repair must fix EVERY listed factual problem.

11. Do not add unrelated facts just to make the script longer.

12. Do not weaken scientifically established facts unnecessarily.

==================================================
STYLE
==================================================

- Strong hook in the first sentence.
- Conversational spoken English.
- One central scientific reveal.
- Fast pacing.
- Clear explanation.
- Punchy ending.
- Natural for text-to-speech.
- No headings.
- No markdown.
- No emojis.
- No stage directions.
- No "subscribe".
- No generic AI filler.
- Do not begin by simply repeating the topic.
- Avoid "Here's something most people don't know".
- Avoid "In conclusion".

Target approximately {target_words} words.

Return ONLY the spoken script.
"""

    response = _send_message_with_retry(
        chat, prompt
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty repair response."
        )

    # Remove accidental markdown fences if Gemini ignores the instruction.
    text = re.sub(
        r"^```(?:text)?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


def validate_and_repair(
    topic: str,
    script: str,
    config: dict,
    target_words: int,
) -> tuple[str, dict]:
    """
    Run the quality gate.

    Flow:

        check
          ↓
        approved? → return
          ↓
        repair
          ↓
        check
          ↓
        approved? → return
          ↓
        repair
          ↓
        final check
          ↓
        return final result
    """

    current_script = script
    last_review = None

    for attempt in range(
        MAX_REPAIR_ATTEMPTS + 1
    ):

        print(
            f"Quality Gate: checking script "
            f"(attempt {attempt + 1}/"
            f"{MAX_REPAIR_ATTEMPTS + 1})..."
        )

        try:
            review = check_script(
                topic,
                current_script,
                config,
                target_words,
            )
        except Exception as e:

            print(
                f"WARNING: Quality Gate check failed ({e})."
            )

            # Do not silently approve if the scientific reviewer failed.
            last_review = {
                "approved": False,
                "score": 0.0,
                "problems": [
                    f"Scientific Quality Gate failed to run: {e}"
                ],
                "corrections": [
                    "Retry the scientific review before publishing."
                ],
                "reason": (
                    "The scientific reviewer did not return a valid result."
                ),
            }

            break

        last_review = review

        print(
            f"Quality Gate: score="
            f"{review.get('score', 0):.1f}/10 "
            f"approved={review.get('approved')}"
        )

        if review.get("problems"):

            for problem in review["problems"][:8]:

                print(
                    f"  - {problem}"
                )

        if review.get("approved"):

            print(
                "Quality Gate: PASSED"
            )

            return (
                current_script,
                review,
            )

        # --------------------------------------------------------
        # If this was the final check, do not repair again.
        # --------------------------------------------------------

        if attempt >= MAX_REPAIR_ATTEMPTS:
            break

        print(
            "Quality Gate: repairing script..."
        )

        try:

            current_script = repair_script(
                topic=topic,
                script=current_script,
                review=review,
                config=config,
                target_words=target_words,
            )

        except Exception as e:

            print(
                f"WARNING: Quality Gate repair failed ({e})."
            )

            # Keep the current script and fail closed.
            break

    print(
        "Quality Gate: FAILED after repair attempts."
    )

    return (
        current_script,
        last_review,
    )
