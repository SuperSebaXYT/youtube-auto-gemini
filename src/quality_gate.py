
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
- Uses Chat.send_message() for FreeInference text model text generation
"""

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.gemini_client import get_client, is_available, TEXT_MODEL as DEFAULT_TEXT_MODEL


# Keep this synchronized with the rest of the project.
TEXT_MODEL = DEFAULT_TEXT_MODEL

# Initial check + 2 repairs = maximum 3 checks.
MAX_REPAIR_ATTEMPTS = 2

# Minimum score required for approval.
MIN_APPROVAL_SCORE = 8.0


class AIProviderUnavailableError(RuntimeError):
    """FreeInference text model remained unavailable after retrying transient API failures."""


def _retry_after_seconds(exc) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    value = headers.get("Retry-After") or headers.get("retry-after")
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _is_retryable_ai_error(exc) -> bool:
    """Recognize transient FreeInference text model/API failures using status fields and message."""
    if isinstance(exc, TimeoutError) or type(exc).__name__ in {
        "APITimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout"
    }:
        return True

    retryable_codes = {408, 409, 429, 500, 502, 503, 504}

    for attribute in ("code", "status_code", "http_status"):
        value = getattr(exc, attribute, None)
        try:
            if int(value) in retryable_codes:
                return True
        except (TypeError, ValueError):
            pass

    status = str(getattr(exc, "status", "")).upper()
    if status in {
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
        "SERVICE_UNAVAILABLE",
        "TOO_MANY_REQUESTS",
    }:
        return True

    message = str(exc).upper()
    if re.search(r"\b(408|409|429|500|502|503|504)\b", message):
        return True

    return any(
        marker in message
        for marker in (
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "SERVICE_UNAVAILABLE",
            "TOO_MANY_REQUESTS",
            "TEMPORARILY OVERLOADED",
            "REQUEST TIMED OUT",
            "TIMED OUT",
            "TIMEOUT",
        )
    )


def _send_message_with_retry(
    chat,
    prompt: str,
    max_attempts: int = 5,
):
    """Retry transient FreeInference text model failures with backoff, jitter and Retry-After."""
    for attempt in range(max_attempts):
        try:
            return chat.send_message(prompt)
        except Exception as exc:
            if not _is_retryable_ai_error(exc):
                raise

            if attempt == max_attempts - 1:
                raise AIProviderUnavailableError(
                    f"FreeInference text model remained unavailable after {max_attempts} attempts: {exc}"
                ) from exc

            backoff = min(30.0, 2.0 * (2 ** attempt))
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = backoff * random.uniform(0.8, 1.2)
            else:
                delay = max(delay, backoff)

            print(
                "WARNING: FreeInference text model transient API failure; "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt + 2}/{max_attempts}): {exc}"
            )
            time.sleep(delay)


def _client():
    """
    Create a FreeInference text model client.
    """

    if not is_available():
        raise RuntimeError("FREEINFERENCE_API_KEY is missing; FreeInference text generation cannot run the Quality Gate.")

    return get_client()


def _chat(client):
    """
    Create a FreeInference text model chat session.

    IMPORTANT:
    Use Chat.send_message() rather than client.models.generate_content()
    for normal text generation.
    """

    return client.chats.create(
        model=TEXT_MODEL
    )


def _clean_json(text: str) -> dict:
    """
    Extract JSON even if FreeInference text model accidentally wraps it in markdown fences.
    """

    if not text:
        raise ValueError(
            "FreeInference text model returned an empty response."
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
            f"FreeInference text model did not return JSON: {text[:500]}"
        )

    try:
        return json.loads(
            text[start:end + 1]
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"FreeInference text model returned invalid JSON: {e}. "
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
    Cheap deterministic checks before spending another FreeInference text model call.

    IMPORTANT:
    This function intentionally does NOT automatically reject normal
    scientific words such as:

        will
        never
        always
        instantly
        completely

    Those words can be scientifically correct in the right context.

    FreeInference text model performs the actual scientific certainty audit.

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
    # FreeInference text model's claim-by-claim scientific audit below is responsible
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
    Ask FreeInference text model to perform a strict scientific/editorial audit.
    """

    basic_problems = _basic_script_checks(
        script, target_words
    )

    client = _client()
    chat = _chat(client)

    prompt = f"""
You are the final scientific fact-checker and editorial gatekeeper for an
animal and wildlife YouTube Shorts channel.

CHANNEL: {config['display_name']}
NICHE: {config['niche']}
TONE: {config['tone']}
TOPIC: {topic}

SCRIPT:
{script}

Your decision determines whether this script may proceed to voice generation
and publication. Factual accuracy is the highest priority.

FACT-CHECK EVERY IMPORTANT CLAIM:
- Confirm the named animal, species, anatomy, behavior, habitat and
  biological mechanism are described accurately.
- Check taxonomy and evolutionary relationships. Do not imply two animals
  are close relatives without evidence.
- Distinguish observed, experimentally supported, inferred, debated and
  hypothetical behavior. Never present uncertainty as fact.
- Check numerical claims, measurements, speeds, strength, lifespan,
  percentages and records. Flag unsupported precision.
- Check cause-and-effect claims and whether context changes the outcome.
- Distinguish species-wide traits from behavior seen only in some individuals,
  populations, sexes, life stages or environments.
- Flag outdated terminology and interpretations.
- Do not invent a correction, source, study or measurement.

ANIMAL-SPECIFIC SAFETY:
- Do not attribute behavior to the wrong species or confuse similar animals.
- Do not exaggerate danger, venom, intelligence, strength, senses or abilities.
- Do not imply deliberate planning unless evidence supports that interpretation.
- Treat rare events and exceptional individuals as exceptions, not universal traits.
- If evidence is mixed, recommend accurate conditional wording.

HOOK AND STORY:
- The first sentence must create interest without making a false promise.
- The payoff must answer the central question.
- Keep the script focused on the animal or wildlife topic.
- It must work as clear spoken narration for a 30-60 second Short.
- Do not reject accurate simplification for a general audience.

APPROVAL:
Approve only when the central claim and supporting claims are accurate,
appropriately qualified and not materially misleading. A major factual error
requires rejection. Entertainment value never offsets a false biological claim.

Score from 1 to 10. Accuracy is a hard constraint; also consider clarity,
curiosity, retention, visual storytelling and natural narration. Do not
penalize words such as "always", "never" or "will" without judging the claim.

Return ONLY valid JSON in exactly this shape:
{{
  "approved": true,
  "score": 9.2,
  "problems": [],
  "corrections": [],
  "reason": "Short explanation"
}}

Use approved=false when a major issue remains. Problems must identify
specific claims. Corrections must state the precise change needed. Do not
invent facts to justify rejection.
"""
    response = _send_message_with_retry(
        chat, prompt
    )

    result = _clean_json(
        response.text
    )

    # ------------------------------------------------------------
    # Normalize FreeInference text model output.
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
    # issues that FreeInference text model should not override.
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

    # If FreeInference text model returned no corrections, explicitly tell the repair model
    # to independently fix the listed problems without inventing science.
    if not corrections:
        corrections = (
            "- Correct every factual problem identified by the fact-check.\n"
            "- If a claim depends on conditions, use scientifically "
            "appropriate conditional wording."
        )

    # Add an explicit, concrete correction when the reviewer flags the
    # recurring whip-spider / pistol-shrimp taxonomy confusion. The generic
    # repair prompt was allowing FreeInference text model to preserve the misleading hook.
    taxonomy_repair = ""
    combined_review = " ".join(
        str(item) for item in review.get("problems", [])
    ) + " " + " ".join(
        str(item) for item in review.get("corrections", [])
    )
    if (
        "pistol shrimp" in combined_review.lower()
        and "whip spider" in (topic + " " + script).lower()
        and any(term in combined_review.lower() for term in ("relative", "taxonom", "phylogen"))
    ):
        taxonomy_repair = """
CRITICAL CORRECTION FOR THIS SCRIPT:
The whip spider (an arachnid) is NOT a close relative of the pistol shrimp
(a crustacean). Do not call it "the pistol shrimp's relative", a cousin,
or imply a close evolutionary relationship. Remove the pistol-shrimp
comparison entirely. Build the hook around the whip spider's own
extraordinarily sensitive, whip-like front legs and its hunting behavior.
The corrected script must not mention the pistol shrimp.
"""

    prompt = f"""
You are a senior animal-science editor repairing a YouTube Short that failed
scientific fact-checking.

CHANNEL: {config['display_name']}
NICHE: {config['niche']}
TOPIC: {topic}

ORIGINAL SCRIPT:
{script}

FACT-CHECK PROBLEMS:
{problems}

REQUIRED CORRECTIONS:
{corrections}

{taxonomy_repair}

Rewrite the narration so every listed factual problem is fixed. Keep the
animal or wildlife premise, but never invent species, anatomy, behavior,
abilities, statistics, observations, studies or discoveries.

BIOLOGICAL ACCURACY:
- Every behavior must belong to the named species.
- Do not invent evolutionary relationships or taxonomic links.
- Distinguish established behavior from rare, disputed or hypothetical cases.
- Do not exaggerate venom, danger, intelligence, strength, senses or speed.
- Use conditional wording when behavior depends on context.
- Preserve accurate facts instead of weakening them unnecessarily.
- Remove unsupported exact numbers rather than guessing.
- Do not add unrelated facts to fill space.

STYLE:
- Strong, accurate first-sentence hook.
- Conversational spoken English.
- One central animal-related reveal.
- Fast pacing and clear explanation.
- Natural for text-to-speech.
- No headings, markdown, emojis, stage directions or calls to subscribe.
- Do not begin by repeating the topic. Avoid generic AI filler.

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
            "FreeInference text model returned an empty repair response."
        )

    # Remove accidental markdown fences if FreeInference text model ignores the instruction.
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
        except AIProviderUnavailableError:
            # Infrastructure failure is not a scientific score. Stop safely
            # and let the next scheduled run retry the review.
            raise

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

        except AIProviderUnavailableError:
            # Do not mislabel a FreeInference text model outage as a failed script repair.
            raise

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
