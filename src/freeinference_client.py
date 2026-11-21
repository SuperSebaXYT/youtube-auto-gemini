"""
Gemini-backed text and vision gateway for Cosmic Curious.

Text generation and image-based Visual QA: Gemini 3.5 Flash Lite.
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import threading
import time
from types import SimpleNamespace

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
TEXT_MODEL = "gemini-3.5-flash-lite"
VISION_MODEL = "gemini-3.5-flash-lite"

# Ordered fallback chain. The primary model remains unchanged.
# Gemini 4 Argon is last because the current free-tier dashboard shows 0/0.
FALLBACK_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-4-argon",
)
MODEL_REQUEST_INTERVALS = {
    "gemini-3.5-flash-lite": 8.0,
    "gemini-3.1-flash-lite": 5.0,
    "gemini-4-hydrogen": 13.0,
    "gemini-3.8-flash": 13.0,
    "gemini-3-flash-preview": 13.0,
    "gemini-2.5-flash": 13.0,
    "gemini-4-argon": 13.0,
}
USAGE_PATH = Path(__file__).resolve().parent.parent / "state" / "gemini_usage.json"

# Shared request gate for every Gemini text and vision call in this process.
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_AT_BY_MODEL = {}


def _retry_after_seconds(exc) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = (headers or {}).get("Retry-After") or (headers or {}).get("retry-after")
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


def _is_rate_limit_error(exc) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if int(status) == 429:
            return True
    except (TypeError, ValueError):
        pass
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc).upper()


def _is_model_unavailable_error(exc) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = str(exc).lower()
    try:
        code = int(status)
    except (TypeError, ValueError):
        code = None
    model_specific_markers = (
        "model", "not found", "unsupported", "not supported",
        "not available", "not enabled", "permission_denied",
    )
    return code in (400, 403, 404) and any(
        marker in message for marker in model_specific_markers
    )


def _paced_completion(client, kwargs):
    """Pace each model independently and fail over on quota/model availability errors."""
    global _LAST_REQUEST_AT_BY_MODEL
    requested_model = kwargs["model"]
    candidates = (requested_model,) + tuple(
        model for model in FALLBACK_MODELS if model != requested_model
    )
    failures = []

    for model in candidates:
        model_kwargs = dict(kwargs)
        model_kwargs["model"] = model
        interval = MODEL_REQUEST_INTERVALS.get(model, 13.0)

        # Exactly one attempt per model. Quota/rate-limit failures immediately
        # move to the next configured fallback; never sleep and retry the same model.
        for attempt in range(1):
            with _REQUEST_LOCK:
                _LAST_REQUEST_AT_BY_MODEL[model] = time.monotonic()
                try:
                    response = client.chat.completions.create(**model_kwargs)
                    if model != requested_model:
                        print(
                            f"Gemini fallback activated: {requested_model} -> {model}."
                        )
                    return response, model
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        failures.append(f"{model}: HTTP 429")
                        if attempt == 0:
                            print(
                                f"Gemini rate limit on {model}; immediately "
                                "trying the next fallback model."
                            )
                            break
                        print(f"Gemini quota exhausted for {model}; trying next model.")
                        break
                    if _is_model_unavailable_error(exc):
                        failures.append(f"{model}: model unavailable")
                        print(f"Gemini model {model} unavailable; trying next model.")
                        break
                    raise

    details = "; ".join(failures)
    raise RuntimeError(
        "All configured Gemini models failed. " + details
        + ". Check Google AI Studio quotas, model access, and billing."
    )

def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return key


def is_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def is_vision_available() -> bool:
    return is_available()


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install dependencies from requirements.txt."
        ) from exc
    return OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=30.0, max_retries=0)


def _record_usage(model: str, usage) -> None:
    if usage is None:
        print(f"Gemini usage unavailable for {model}.")
        return

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    today = datetime.now(timezone.utc).date().isoformat()

    try:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = {}
        if data.get("date") != today:
            data = {"date": today, "requests": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "total_tokens": 0, "by_model": {}}
        data["requests"] = int(data.get("requests", 0)) + 1
        data["prompt_tokens"] = int(data.get("prompt_tokens", 0)) + prompt_tokens
        data["completion_tokens"] = int(data.get("completion_tokens", 0)) + completion_tokens
        data["total_tokens"] = int(data.get("total_tokens", 0)) + total_tokens
        model_data = data.setdefault("by_model", {}).setdefault(
            model, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        model_data["requests"] += 1
        model_data["prompt_tokens"] += prompt_tokens
        model_data["completion_tokens"] += completion_tokens
        model_data["total_tokens"] += total_tokens
        temporary = USAGE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(USAGE_PATH)
    except OSError as exc:
        print(f"WARNING: could not persist Gemini usage counters: {exc}")

    print(
        f"Gemini usage [{model}]: input={prompt_tokens}, "
        f"output={completion_tokens}, total={total_tokens} tokens."
    )


def _complete(prompt: str, *, model: str, max_output_tokens: int | None = None,
              json_mode: bool = False, images: list[tuple[bytes, str]] | None = None):
    # Route both text and image requests through Gemini. Caller-supplied model
    # names cannot change the configured model for either modality.
    model = VISION_MODEL if images else TEXT_MODEL
    if not prompt.strip():
        raise RuntimeError("AI request has no text prompt.")

    content = (
        "SYSTEM CONSTRAINT: Do not call or request tools, web search, browser, shell, "
        "or external functions. Answer directly using the supplied prompt and context only.\n\n"
        + prompt
    )
    if json_mode:
        content += "\n\nReturn only one valid JSON object. Do not use markdown fences or commentary."

    if images:
        message_content = [{"type": "text", "text": content}]
        for image_bytes, mime_type in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            })
    else:
        message_content = content

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
    }
    if max_output_tokens:
        kwargs["max_tokens"] = int(max_output_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response, actual_model = _paced_completion(_client(), kwargs)
    _record_usage(actual_model, getattr(response, "usage", None))
    if not response.choices:
        raise RuntimeError(f"Gemini returned no choices for {model}.")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"Gemini returned an empty response for {model}.")
    return SimpleNamespace(
        text=text,
        choices=response.choices,
        usage=getattr(response, "usage", None),
        raw=response,
    )


def _validate_json_output(content: str) -> None:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a complete JSON object.")
    json.loads(candidate[start:end + 1])


class _ChatSession:
    def __init__(self, owner, model: str, config=None):
        self.owner = owner
        self.model = model or TEXT_MODEL
        self.config = config

    def send_message(self, prompt: str):
        max_tokens = getattr(self.config, "max_output_tokens", None)
        json_mode = getattr(self.config, "response_mime_type", None) == "application/json"
        prompt_lower = str(prompt).lower()
        if "return only valid json" in prompt_lower or "return only one valid json object" in prompt_lower:
            json_mode = True
        response = _complete(
            str(prompt), model=TEXT_MODEL, max_output_tokens=max_tokens, json_mode=json_mode
        )
        if json_mode:
            _validate_json_output(response.text)
        return SimpleNamespace(
            text=response.text,
            candidates=[SimpleNamespace(finish_reason=response.choices[0].finish_reason)],
            raw=response.raw,
        )


class _Chats:
    def __init__(self, owner):
        self.owner = owner

    def create(self, model=None, config=None):
        return _ChatSession(self.owner, model or TEXT_MODEL, config)


class _Models:
    def __init__(self, owner):
        self.owner = owner

    def generate_content(self, model=None, contents=None):
        prompt_parts = []
        images = []
        for item in (contents or []):
            if isinstance(item, str):
                prompt_parts.append(item)
            else:
                data = getattr(item, "data", None)
                mime = getattr(item, "mime_type", None) or "image/jpeg"
                if data:
                    images.append((data, mime))
        if images:
            response = generate_vision("\n".join(prompt_parts), images)
            return SimpleNamespace(text=response.text, raw=response.raw)
        return generate_response("\n".join(prompt_parts))


class GeminiClient:
    def __init__(self):
        _api_key()
        self.chats = _Chats(self)
        self.models = _Models(self)


# Compatibility name: existing pipeline modules import get_client().


def get_client():
    return GeminiClient()


def generate(prompt: str, max_output_tokens: int | None = None,
             model: str = TEXT_MODEL, json_mode: bool = False) -> str:
    response = _complete(
        prompt, model=TEXT_MODEL, max_output_tokens=max_output_tokens, json_mode=json_mode
    )
    if json_mode:
        _validate_json_output(response.text)
    return response.text


def generate_vision(prompt: str, images: list[tuple[bytes, str]], model: str | None = None):
    if not images:
        raise RuntimeError("Visual verification requires at least one image.")
    return _complete(
        prompt, model=VISION_MODEL, max_output_tokens=20, images=images
    )


def generate_response(prompt: str, max_output_tokens: int | None = None,
                      model: str = TEXT_MODEL):
    response = _complete(prompt, model=TEXT_MODEL, max_output_tokens=max_output_tokens)
    return SimpleNamespace(
        text=response.text,
        candidates=[SimpleNamespace(finish_reason=response.choices[0].finish_reason)],
        raw=response.raw,
    )
