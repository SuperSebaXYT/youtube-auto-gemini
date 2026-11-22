"""Wikipedia-first curiosity discovery for Cosmic Curious.

Gemini proposes research leads; the English Wikipedia API supplies the
evidence; Gemini can only select a topic from the returned article extracts.
"""
from __future__ import annotations

import json
import re
import requests

from src.freeinference_client import generate as gemini_generate

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "CosmicCuriousBot/1.0 (animal-facts research; https://github.com/SuperSebaXYT/youtube-auto-gemini)"
MAX_RESEARCH_CANDIDATES = 8


def _json_object(text: str) -> dict:
    """Extract the first complete JSON object from Gemini output."""
    value = str(text or "").strip().lstrip("\ufeff")
    value = re.sub(r"^\s*\x60{3}[a-zA-Z0-9_-]*\s*", "", value)
    start = value.find("{")
    if start < 0:
        raise ValueError("Gemini did not return a JSON object for Wikipedia research.")
    try:
        data, _end = json.JSONDecoder().raw_decode(value[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini Wikipedia response did not contain a valid complete JSON object: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("Wikipedia research response is not a JSON object.")
    return data

def _wiki_get(params: dict) -> dict:
    response = requests.get(
        WIKIPEDIA_API,
        params={**params, "format": "json", "utf8": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Wikipedia API returned an invalid response.")
    return payload


def _species_key(value: str) -> str:
    value = str(value or "").casefold().strip()
    match = re.search(r"\(([a-z][a-z-]+\s+[a-z][a-z-]+)\)", value)
    if match:
        value = match.group(1)
    return re.sub(r"[^a-z0-9]+", "", value)


def _topic_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _is_duplicate_topic(topic: str, used_topics: list[str]) -> bool:
    from difflib import SequenceMatcher
    candidate = _topic_key(topic)
    if not candidate:
        return True
    candidate_words = set(candidate.split())
    for old in used_topics:
        previous = _topic_key(old)
        if not previous:
            continue
        if candidate == previous or SequenceMatcher(None, candidate, previous).ratio() >= 0.82:
            return True
        previous_words = set(previous.split())
        overlap = len(candidate_words & previous_words) / max(1, min(len(candidate_words), len(previous_words)))
        if overlap >= 0.82:
            return True
    return False


def discover_wikipedia_evidence(
    winner_context: str,
    used_animals: list[str],
    used_topics: list[str],
) -> list[dict]:
    """Let Gemini discover leads, then retrieve supporting Wikipedia extracts."""
    prompt = f"""
You are a wildlife curiosity researcher. Discover 8 genuinely surprising,
specific, well-documented animal facts worth turning into YouTube Shorts.

Do not write scripts and do not invent facts. Propose research leads only.
Explore unusual mechanisms, strange adaptations, counterintuitive behavior,
extreme survival, animal senses, communication, and unexpected biology.
Avoid generic facts and avoid species already used by this channel.

CHANNEL PERFORMANCE EVIDENCE:
{winner_context}

NEVER-USED SPECIES RULE. These species are permanently excluded:
{json.dumps(used_animals, ensure_ascii=False)}

Previously used topic premises to avoid:
{json.dumps(used_topics[-120:], ensure_ascii=False)}

Return JSON only:
{{"candidates":[
  {{"animal":"common species name","scientific_name":"Genus species",
    "curiosity_angle":"specific unusual behavior or adaptation",
    "wikipedia_query":"short English Wikipedia search phrase"}}
]}}

Every candidate must be one exact species, not a family or broad group.
Do not state unsupported claims as facts. Search phrase should help find
the species article or a directly relevant Wikipedia article.
"""
    raw = gemini_generate(prompt, max_output_tokens=2200, json_mode=True)
    data = _json_object(raw)
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Gemini returned no Wikipedia research candidates.")

    evidence = []
    seen_species = set()
    for item in candidates[:MAX_RESEARCH_CANDIDATES]:
        if not isinstance(item, dict):
            continue
        animal = str(item.get("animal") or "").strip()
        scientific = str(item.get("scientific_name") or "").strip()
        angle = str(item.get("curiosity_angle") or "").strip()
        query = str(item.get("wikipedia_query") or "").strip()
        if not animal or not scientific or not angle or not query:
            continue
        if not re.fullmatch(r"[A-Z][a-z-]+\s+[a-z][a-z-]+", scientific):
            continue
        key = scientific.casefold()
        if key in seen_species:
            continue
        # Enforce the permanent no-repeat rule in code, not merely in Gemini's prompt.
        used_species = {_species_key(value) for value in used_animals}
        if _species_key(scientific) in used_species or _species_key(animal) in used_species:
            print(f"Wikipedia Research: skipping previously used species {animal} ({scientific}).")
            continue
        seen_species.add(key)

        try:
            search_data = _wiki_get({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
            })
            hits = search_data.get("query", {}).get("search", [])
            for hit in hits[:2]:
                page_title = str(hit.get("title") or "").strip()
                if not page_title:
                    continue
                page_data = _wiki_get({
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,
                    "exchars": 4500,
                    "titles": page_title,
                })
                pages = page_data.get("query", {}).get("pages", {})
                page = next(iter(pages.values()), {})
                extract = re.sub(r"\s+", " ", str(page.get("extract") or "")).strip()
                if not extract or page.get("missing") is not None:
                    continue
                identity_terms = (animal.casefold(), scientific.casefold())
                searchable = (page_title + " " + extract).casefold()
                if not any(term in searchable for term in identity_terms):
                    continue
                evidence.append({
                    "animal": animal,
                    "scientific_name": scientific,
                    "curiosity_angle": angle,
                    "article_title": page_title,
                    "article_url": "https://en.wikipedia.org/wiki/" + page_title.replace(" ", "_"),
                    "extract": extract[:4500],
                })
                break
        except Exception as exc:
            print(f"Wikipedia research lookup failed for {animal} ({exc}); skipping candidate.")

    if not evidence:
        raise RuntimeError(
            "Gemini found no animal curiosity leads with matching English Wikipedia evidence. "
            "Topic selection is blocked rather than inventing facts."
        )
    print(
        f"Wikipedia Research: verified article extracts for "
        f"{len(evidence)} of {len(candidates[:MAX_RESEARCH_CANDIDATES])} Gemini-discovered candidates."
    )
    return evidence


def choose_wikipedia_topic(evidence: list[dict], winner_context: str,
                           used_animals: list[str], used_topics: list[str]) -> dict:
    """Select a verified curiosity; reject duplicates and try another species."""
    eligible = [
        item for item in evidence
        if _species_key(item.get("scientific_name", "")) not in
        {_species_key(value) for value in used_animals}
        and _species_key(item.get("animal", "")) not in
        {_species_key(value) for value in used_animals}
    ]
    rejected = []
    max_attempts = min(len(eligible), 8)

    for attempt in range(max_attempts):
        remaining = [item for item in eligible if item not in rejected]
        if not remaining:
            break
        compact = [{
            "animal": x["animal"],
            "scientific_name": x["scientific_name"],
            "curiosity_angle": x["curiosity_angle"],
            "article_title": x["article_title"],
            "article_url": x["article_url"],
            "wikipedia_extract": x["extract"],
        } for x in remaining]
        prompt = f"""
Choose the single strongest, most surprising, visually explainable animal
curiosity for a 35-55 second global-audience YouTube Short.

Use channel performance data to judge broad appeal and hook potential:
{winner_context}

The following are the ONLY eligible species and factual source material.
Every factual claim in the selected topic must be supported by its Wikipedia
extract. Do not add outside knowledge or infer details not stated there.
{json.dumps(compact, ensure_ascii=False)}

Permanently excluded species:
{json.dumps(used_animals, ensure_ascii=False)}
Previously used topic premises:
{json.dumps(used_topics[-120:], ensure_ascii=False)}
Previously rejected selections in this run:
{json.dumps([{ "animal": x["animal"], "scientific_name": x["scientific_name"], "angle": x["curiosity_angle"] } for x in rejected], ensure_ascii=False)}

Choose one exact species and one specific curiosity. Do not choose a generic
species overview. Do not exaggerate. Return JSON only:
{{"animal":"exact common name","scientific_name":"Genus species",
  "topic":"specific compelling topic phrased as a clear curiosity",
  "source_article_title":"exact article title",
  "source_article_url":"exact article URL",
  "verified_evidence":"2-4 concise sentences paraphrased only from the extract"}}
"""
        raw = gemini_generate(prompt, max_output_tokens=1100, json_mode=True)
        selected = _json_object(raw)
        required = ("animal", "scientific_name", "topic", "source_article_title",
                    "source_article_url", "verified_evidence")
        if any(not str(selected.get(key) or "").strip() for key in required):
            raise ValueError("Gemini selected a topic without complete Wikipedia evidence metadata.")
        match = next((
            item for item in remaining
            if item["animal"].casefold() == str(selected["animal"]).casefold()
            and item["scientific_name"].casefold() == str(selected["scientific_name"]).casefold()
            and item["article_title"] == str(selected["source_article_title"])
        ), None)
        if match is None:
            raise ValueError("Gemini selected a species or article outside the verified Wikipedia candidates.")

        selected_species = _species_key(match["scientific_name"])
        if selected_species in {_species_key(value) for value in used_animals}:
            rejected.append(match)
            continue
        topic = str(selected["topic"]).strip()
        if _is_duplicate_topic(topic, used_topics):
            print(f"Wikipedia Research: duplicate topic rejected; trying another verified species: {topic}")
            rejected.append(match)
            continue

        selected["source_article_url"] = match["article_url"]
        return selected

    raise RuntimeError(
        "Wikipedia Research exhausted verified unused species: every selected candidate "
        "was already used or had a duplicate topic. Publishing is blocked."
    )
