"""
editorial_pass.py
Runs the hook/SEO/thumbnail/retention agents as one editorial polish
stage, after the script has already passed quality_gate's science
fact-check. Each agent call is independently fail-safe: any failure
(rate limit, bad JSON, empty result) falls back to the existing value
and prints a warning, never blocks the pipeline.

Ordering matters:
1. Retention pass runs first, since it may rewrite the script body.
   Its own prompt instructs "without weakening scientific accuracy",
   but as a cheap extra safeguard we reject a rewrite whose word count
   swings wildly from the original (a garbled/degenerate rewrite is far
   more likely to have a wrong word count than a genuine pacing fix).
   The main pipeline runs a final quality_gate audit after this entire
   editorial pass, so any rewritten hook/body is checked before TTS.
2. Hook pass runs second, on the (possibly retention-updated) script --
   replaces just the opening sentence with the highest-scored hook,
   leaving the rest of the script body untouched.
3. SEO pass supplies a recommended title + extra tags/hashtags, merged
   with (not replacing) what build_description() already generates.
4. Thumbnail pass supplies banner text for generate_thumbnail(), with a
   length guard since it's agent-generated text going straight onto a
   fixed-size banner.
"""
import re


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _apply_retention_pass(topic: str, script: str) -> str:
    from src.agents.retention_agent import review_script
    review = review_script(topic, script)
    improved = review.get("improved_script") if isinstance(review, dict) else None
    if not improved or not improved.strip():
        return script

    original_words, new_words = _word_count(script), _word_count(improved)
    if original_words == 0:
        return script
    ratio = new_words / original_words
    if not (0.6 <= ratio <= 1.6):
        print(f"WARNING: retention_agent rewrite word count swung too far "
              f"({original_words} -> {new_words} words); keeping original script.")
        return script

    print(f"Retention pass: adopted improved script (score={review.get('score', '?')}).")
    return improved.strip()


def _apply_hook_pass(topic: str, script: str) -> str:
    from src.agents.hook_agent import generate_hooks
    hooks = generate_hooks(topic, script, count=8)
    if not hooks:
        return script

    def _score(h):
        return sum(h.get(k, 0) for k in ("clarity", "curiosity", "retention", "science_safety"))

    best = max(hooks, key=_score)
    best_hook = (best.get("hook") or "").strip()
    if not best_hook:
        return script

    # Replace only the opening sentence (up to the first ./!/?), keep the
    # rest of the script body exactly as fact-checked.
    match = re.search(r"[.!?]\s+", script)
    if not match:
        return script  # single-sentence script -- too risky to touch, skip
    rest_of_script = script[match.end():]
    print(f"Hook pass: replaced opening with top-scored hook (score={_score(best)}).")
    return f"{best_hook} {rest_of_script}"


def _apply_seo_pass(topic: str, script: str, title: str) -> tuple:
    from src.agents.seo_agent import optimize
    result = optimize(topic, script)
    if not isinstance(result, dict):
        return title, [], []

    new_title = (result.get("recommended_title") or "").strip()
    if new_title and len(new_title) <= 100:
        title = new_title

    extra_tags = [t for t in result.get("tags", []) if isinstance(t, str) and t.strip()]
    extra_hashtags = [h for h in result.get("hashtags", []) if isinstance(h, str) and h.strip()]
    return title, extra_tags, extra_hashtags


def _apply_thumbnail_pass(topic: str, script: str) -> str:
    from src.agents.thumbnail_agent import concepts
    options = concepts(topic, script, count=5)
    if not options:
        return None

    def _score(c):
        return c.get("curiosity", 0) + c.get("clarity", 0)

    best = max(options, key=_score)
    text = (best.get("text") or "").strip()
    if not text or len(text) > 60:  # a 60+ char banner text won't fit/read well
        return None
    return text


def run_editorial_pass(topic: str, script: str, title: str, config: dict) -> dict:
    """Main entry point. Returns a dict with (possibly updated) script,
    title, extra_tags, extra_hashtags, and thumbnail_text. Every field
    falls back to a safe default (original script/title, empty lists,
    None) if its agent call fails for any reason."""
    if not config.get("use_editorial_agents"):
        return {"script": script, "title": title, "extra_tags": [],
                "extra_hashtags": [], "thumbnail_text": None}

    try:
        script = _apply_retention_pass(topic, script)
    except Exception as e:
        print(f"WARNING: retention_agent failed ({e}); keeping original script.")

    try:
        script = _apply_hook_pass(topic, script)
    except Exception as e:
        print(f"WARNING: hook_agent failed ({e}); keeping original opening.")

    extra_tags, extra_hashtags = [], []
    try:
        title, extra_tags, extra_hashtags = _apply_seo_pass(topic, script, title)
    except Exception as e:
        print(f"WARNING: seo_agent failed ({e}); keeping original title/tags.")

    thumbnail_text = None
    try:
        thumbnail_text = _apply_thumbnail_pass(topic, script)
    except Exception as e:
        print(f"WARNING: thumbnail_agent failed ({e}); thumbnail will use auto-derived text.")

    return {
        "script": script, "title": title, "extra_tags": extra_tags,
        "extra_hashtags": extra_hashtags, "thumbnail_text": thumbnail_text,
    }
