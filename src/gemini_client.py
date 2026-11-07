"""
FreeInference-backed text and vision gateway for Cosmic Curious.

Text generation: DeepSeek V4 Flash.
Visual QA: Qwen3.6-35B (vision-capable).
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

BASE_URL = os.environ.get("FREEINFERENCE_BASE_URL", "https://freeinference.org/v1").rstrip("/")
TEXT_MODEL = "deepseek-v4-flash"
VISION_MODEL = "qwen3.6-35b"
USAGE_PATH = Path(__file__).resolve().parent.parent / "state" / "freeinference_usage.json"


def _api_key() -> str:
    key = os.environ.get("FREEINFERENCE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FREEINFERENCE_API_KEY is not set.")
    return key


def is_available() -> bool:
    return bool(os.environ.get("FREEINFERENCE_API_KEY", "").strip())


def is_vision_available() -> bool:
    return is_available()


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install dependencies from requirements.txt."
        ) from exc
    return OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=120.0, max_retries=0)


def _record_usage(model: str, usage) -> None:
    if usage is None:
        print(f"FreeInference usage unavailable for {model}.")
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
        print(f"WARNING: could not persist FreeInference usage counters: {exc}")

    print(
        f"FreeInference usage [{model}]: input={prompt_tokens}, "
        f"output={completion_tokens}, total={total_tokens} tokens."
    )


def _complete(prompt: str, *, model: str, max_output_tokens: int | None = None,
              json_mode: bool = False, images: list[tuple[bytes, str]] | None = None):
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

    response = _client().chat.completions.create(**kwargs)
    _record_usage(model, getattr(response, "usage", None))
    if not response.choices:
        raise RuntimeError(f"FreeInference returned no choices for {model}.")
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"FreeInference returned an empty response for {model}.")
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


class FreeInferenceClient:
    def __init__(self):
        _api_key()
        self.chats = _Chats(self)
        self.models = _Models(self)


# Compatibility name: existing pipeline modules import get_client().


def get_client():
    return FreeInferenceClient()


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
        prompt, model=VISION_MODEL, max_output_tokens=1200, images=images
    )


def generate_response(prompt: str, max_output_tokens: int | None = None,
                      model: str = TEXT_MODEL):
    response = _complete(prompt, model=TEXT_MODEL, max_output_tokens=max_output_tokens)
    return SimpleNamespace(
        text=response.text,
        candidates=[SimpleNamespace(finish_reason=response.choices[0].finish_reason)],
        raw=response.raw,
    )
