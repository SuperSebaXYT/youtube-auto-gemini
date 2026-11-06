"""
Shared text and vision client for Gemini and Groq.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_OUTPUT_WINDOW_SECONDS = 60.0
GROQ_OUTPUT_BUDGET_PER_MINUTE = 900
GROQ_INPUT_BUDGET_PER_MINUTE = 6500
GROQ_COMBINED_BUDGET_PER_MINUTE = 7500
GROQ_TEXT_OUTPUT_CAP = 450
GROQ_VISION_OUTPUT_CAP = 180

# Approximate Groq's rolling per-minute input/output budgets. Reserve before
# each call so the pipeline spaces requests instead of repeatedly receiving 429.
_GROQ_TOKEN_EVENTS = deque()
_GROQ_OUTPUT_LOCK = threading.Lock()


def _estimate_groq_input_tokens(messages) -> int:
    text_characters = 0
    image_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_characters += len(content)
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_characters += len(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    image_count += 1
    # Base64 image data is deliberately not counted as text.
    return max(1, (text_characters + 3) // 4 + image_count * 1024)


def _reserve_groq_budget(messages, output_tokens: int) -> None:
    input_tokens = _estimate_groq_input_tokens(messages)
    output_tokens = max(1, int(output_tokens))
    while True:
        with _GROQ_OUTPUT_LOCK:
            now = time.monotonic()
            while (
                _GROQ_TOKEN_EVENTS
                and now - _GROQ_TOKEN_EVENTS[0][0] >= GROQ_OUTPUT_WINDOW_SECONDS
            ):
                _GROQ_TOKEN_EVENTS.popleft()

            input_used = sum(event[1] for event in _GROQ_TOKEN_EVENTS)
            output_used = sum(event[2] for event in _GROQ_TOKEN_EVENTS)
            combined_used = input_used + output_used
            fits = (
                input_used + input_tokens <= GROQ_INPUT_BUDGET_PER_MINUTE
                and output_used + output_tokens <= GROQ_OUTPUT_BUDGET_PER_MINUTE
                and combined_used + input_tokens + output_tokens
                    <= GROQ_COMBINED_BUDGET_PER_MINUTE
            )
            if fits:
                _GROQ_TOKEN_EVENTS.append((now, input_tokens, output_tokens))
                return

            oldest_at = _GROQ_TOKEN_EVENTS[0][0]
            delay = max(0.1, GROQ_OUTPUT_WINDOW_SECONDS - (now - oldest_at) + 0.1)

        print(
            f"Groq token budget reached; waiting {delay:.1f}s "
            "before the next request."
        )
        time.sleep(delay)
GROQ_USER_AGENT = (
    "CosmicCurious/1.0 "
    "(GitHub Actions; https://github.com/SuperSebaXYT/youtube-auto-gemini)"
)


def _provider() -> str:
    return os.environ.get("AI_PROVIDER", "gemini").strip().lower()


def _model(model: str | None = None) -> str:
    if _provider() in {"groq"}:
        return os.environ.get("GROQ_TEXT_MODEL") or os.environ.get("GROQ_MODEL") or os.environ.get("AI_MODEL") or model or GROQ_MODEL
    return model or GEMINI_MODEL


def is_available() -> bool:
    if _provider() in {"groq"}:
        return bool(os.environ.get("GROQ_API_KEY"))
    return bool(os.environ.get("GEMINI_API_KEY"))


def _groq_request(messages, max_output_tokens: int | None, model: str | None = None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    selected_model = _model(model)
    body = {
        "model": selected_model,
        "messages": messages,
    }
    is_vision = any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in message["content"]
        )
        for message in messages
    )
    output_cap = GROQ_VISION_OUTPUT_CAP if is_vision else GROQ_TEXT_OUTPUT_CAP
    requested_tokens = max_output_tokens if max_output_tokens else output_cap
    completion_budget = min(int(requested_tokens), output_cap)
    body["max_completion_tokens"] = completion_budget

    if selected_model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
        body["include_reasoning"] = False
    elif selected_model.startswith("qwen/qwen3.8"):
        body["reasoning_effort"] = "none"

    _reserve_groq_budget(messages, completion_budget)

    request = Request(
        GROQ_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": GROQ_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error = RuntimeError(f"Groq API HTTP {exc.code}: {detail}")
        # Preserve Retry-After so the shared retry logic can respect Groq's
        # rate-limit window instead of retrying too early.
        error.response = SimpleNamespace(headers=exc.headers)
        raise error from exc
    except URLError as exc:
        raise RuntimeError(f"Groq API connection failed: {exc}") from exc

    try:
        choice = payload["choices"][0]
        message = choice["message"]
        text = message.get("content", "")
        if isinstance(text, list):
            text = "".join(
                part.get("text", "")
                for part in text
                if isinstance(part, dict)
            )
        text = str(text or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Groq returned an unexpected response: {str(payload)[:500]}"
        ) from exc

    if not text:
        raise RuntimeError("Groq returned an empty response.")

    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(finish_reason=choice.get("finish_reason"))
        ],
        raw=payload,
    )


class _GroqChat:
    def __init__(self, model: str | None = None, max_output_tokens: int | None = None):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.messages = []

    def send_message(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = _groq_request(
                self.messages,
                self.max_output_tokens,
                self.model,
            )
        except Exception:
            self.messages.pop()
            raise
        self.messages.append(
            {"role": "assistant", "content": response.text}
        )
        return response


class _GroqChats:
    def create(self, model: str | None = None, config=None, max_output_tokens=None):
        if max_output_tokens is None and config is not None:
            max_output_tokens = getattr(config, "max_output_tokens", None)
        return _GroqChat(model=model, max_output_tokens=max_output_tokens)


class _GroqClient:
    def __init__(self):
        self.chats = _GroqChats()


def get_client():
    if _provider() in {"groq"}:
        if not is_available():
            raise RuntimeError("GROQ_API_KEY is not set")
        return _GroqClient()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai
    return genai.Client(api_key=api_key)


def generate(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = GEMINI_MODEL,
) -> str:
    """Generate text using the provider selected by AI_PROVIDER."""
    client = get_client()
    if _provider() in {"groq"}:
        response = client.chats.create(
            model=_model(model),
            max_output_tokens=max_output_tokens,
        ).send_message(prompt)
    else:
        from google.genai import types
        if max_output_tokens:
            config = types.GenerateContentConfig(
                max_output_tokens=max_output_tokens
            )
            chat = client.chats.create(model=model, config=config)
        else:
            chat = client.chats.create(model=model)
        response = chat.send_message(prompt)

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError(f"{_provider()} returned an empty response.")
    return text.strip()


def generate_vision(
    prompt: str,
    images: list[tuple[bytes, str]],
    model: str | None = None,
):
    """Inspect image bytes and return a text response."""
    if _provider() in {"groq"}:
        content = [{"type": "text", "text": prompt}]
        for image_bytes, mime_type in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}"
                },
            })
        return _groq_request(
            [{"role": "user", "content": content}],
            max_output_tokens=GROQ_VISION_OUTPUT_CAP,
            model=(
                model
                or os.environ.get("GROQ_VISION_MODEL")
                or GROQ_VISION_MODEL
            ),
        )

    client = get_client()
    from google.genai import types
    contents = [prompt]
    for image_bytes, mime_type in images:
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        )
    return client.models.generate_content(
        model=model or GEMINI_MODEL,
        contents=contents,
    )


def generate_response(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = GEMINI_MODEL,
):
    """Generate text and return the complete provider response."""
    client = get_client()
    if _provider() in {"groq"}:
        return client.chats.create(
            model=_model(model),
            max_output_tokens=max_output_tokens,
        ).send_message(prompt)

    from google.genai import types
    if max_output_tokens:
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens
        )
        chat = client.chats.create(model=model, config=config)
    else:
        chat = client.chats.create(model=model)
    return chat.send_message(prompt)
