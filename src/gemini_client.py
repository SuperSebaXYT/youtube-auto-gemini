"""
Shared text and vision client for Gemini and xAI Grok.
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
XAI_MODEL = "grok-4.7"
XAI_URL = "https://api.x.ai/v1/chat/completions"


def _provider() -> str:
    return os.environ.get("AI_PROVIDER", "gemini").strip().lower()


def _model(model: str | None = None) -> str:
    if _provider() in {"grok", "xai"}:
        return os.environ.get("XAI_MODEL") or os.environ.get("AI_MODEL") or model or XAI_MODEL
    return model or GEMINI_MODEL


def is_available() -> bool:
    if _provider() in {"grok", "xai"}:
        return bool(os.environ.get("XAI_API_KEY"))
    return bool(os.environ.get("GEMINI_API_KEY"))


def _grok_request(messages, max_output_tokens: int | None, model: str | None = None):
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set")

    body = {
        "model": _model(model),
        "messages": messages,
    }
    if max_output_tokens:
        body["max_tokens"] = max_output_tokens

    request = Request(
        XAI_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"xAI API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"xAI API connection failed: {exc}") from exc

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
            f"xAI returned an unexpected response: {str(payload)[:500]}"
        ) from exc

    if not text:
        raise RuntimeError("xAI returned an empty response.")

    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(finish_reason=choice.get("finish_reason"))
        ],
        raw=payload,
    )


class _GrokChat:
    def __init__(self, model: str | None = None, max_output_tokens: int | None = None):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.messages = []

    def send_message(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = _grok_request(
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


class _GrokChats:
    def create(self, model: str | None = None, config=None, max_output_tokens=None):
        if max_output_tokens is None and config is not None:
            max_output_tokens = getattr(config, "max_output_tokens", None)
        return _GrokChat(model=model, max_output_tokens=max_output_tokens)


class _GrokClient:
    def __init__(self):
        self.chats = _GrokChats()


def get_client():
    if _provider() in {"grok", "xai"}:
        if not is_available():
            raise RuntimeError("XAI_API_KEY is not set")
        return _GrokClient()

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
    if _provider() in {"grok", "xai"}:
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
    if _provider() in {"grok", "xai"}:
        content = [{"type": "text", "text": prompt}]
        for image_bytes, mime_type in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}"
                },
            })
        return _grok_request(
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
    if _provider() in {"grok", "xai"}:
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
