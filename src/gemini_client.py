"""
Shared text and vision client for Gemini and Groq.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_USER_AGENT = (
    "CosmicCurious/1.0 "
    "(GitHub Actions; https://github.com/SuperSebaXYT/youtube-auto-gemini)"
)


def _provider() -> str:
    return os.environ.get("AI_PROVIDER", "gemini").strip().lower()


def _model(model: str | None = None) -> str:
    if _provider() in {"groq"}:
        return os.environ.get("GROQ_MODEL") or os.environ.get("AI_MODEL") or model or GROQ_MODEL
    return model or GEMINI_MODEL


def is_available() -> bool:
    if _provider() in {"groq"}:
        return bool(os.environ.get("GROQ_API_KEY"))
    return bool(os.environ.get("GEMINI_API_KEY"))


def _groq_request(messages, max_output_tokens: int | None, model: str | None = None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    body = {
        "model": _model(model),
        "messages": messages,
    }
    # Groq's free tier has a low per-minute output-token budget. Keep
    # individual generations bounded and leave room for subsequent QA calls.
    requested_tokens = max_output_tokens if max_output_tokens else 700
    body["max_completion_tokens"] = min(int(requested_tokens), 900)

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
            max_output_tokens=1200,
            model=model,
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
