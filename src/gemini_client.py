"""
Shared text and vision client for Gemini and OpenRouter.
"""

from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GEMINI_MODEL = "gemini-3.5-flash-lite"
OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _provider() -> str:
    return os.environ.get("AI_PROVIDER", "gemini").strip().lower()


def _model(model: str | None = None) -> str:
    if _provider() == "openrouter":
        return (
            os.environ.get("OPENROUTER_MODEL")
            or os.environ.get("AI_MODEL")
            or OPENROUTER_MODEL
        )
    return model or GEMINI_MODEL


def is_available() -> bool:
    if _provider() == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY"))
    return bool(os.environ.get("GEMINI_API_KEY"))


def _gemini_fallback_request(messages, max_output_tokens: int | None):
    """Replay an OpenRouter request through Gemini when OpenRouter is unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; no AI fallback is available")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = []

    for message in messages:
        role = "model" if message.get("role") == "assistant" else "user"
        raw_content = message.get("content", "")
        parts = []

        if isinstance(raw_content, str):
            if raw_content:
                parts.append(types.Part.from_text(text=raw_content))
        elif isinstance(raw_content, list):
            for item in raw_content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(types.Part.from_text(text=text))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:") and "," in image_url:
                        header, encoded = image_url.split(",", 1)
                        mime_type = header[5:].split(";", 1)[0]
                        parts.append(types.Part.from_bytes(
                            data=base64.b64decode(encoded),
                            mime_type=mime_type,
                        ))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    if not contents:
        raise RuntimeError("No usable prompt content for Gemini fallback")

    config = None
    if max_output_tokens:
        config = types.GenerateContentConfig(max_output_tokens=max_output_tokens)

    return client.models.generate_content(
        model=os.environ.get("GEMINI_FALLBACK_MODEL", GEMINI_MODEL),
        contents=contents,
        config=config,
    )


def _openrouter_request_with_fallback(messages, max_output_tokens: int | None, model: str | None = None):
    try:
        return _openrouter_request(messages, max_output_tokens, model)
    except RuntimeError as openrouter_error:
        if not os.environ.get("GEMINI_API_KEY"):
            raise

        print("[AI] OpenRouter failed; retrying the request through Gemini fallback.")
        try:
            response = _gemini_fallback_request(messages, max_output_tokens)
        except Exception as fallback_error:
            raise RuntimeError(
                "OpenRouter failed and Gemini fallback also failed: "
                f"{fallback_error}"
            ) from fallback_error

        if not getattr(response, "text", None):
            raise RuntimeError(
                "OpenRouter failed and Gemini fallback returned an empty response."
            ) from openrouter_error
        return response


def _openrouter_request(messages, max_output_tokens: int | None, model: str | None = None):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    body = {
        "model": _model(model),
        "messages": messages,
    }
    if max_output_tokens:
        body["max_tokens"] = max_output_tokens

    request = Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SuperSebaXYT/youtube-auto-gemini",
            "X-Title": "Cosmic Curious",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter API connection failed: {exc}") from exc

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
            f"OpenRouter returned an unexpected response: {str(payload)[:500]}"
        ) from exc

    if not text:
        raise RuntimeError("OpenRouter returned an empty response.")

    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(finish_reason=choice.get("finish_reason"))
        ],
        raw=payload,
    )


class _OpenRouterChat:
    def __init__(self, model: str | None = None, max_output_tokens: int | None = None):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.messages = []

    def send_message(self, prompt):
        self.messages.append({"role": "user", "content": prompt})
        try:
            response = _openrouter_request_with_fallback(
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


class _OpenRouterChats:
    def create(self, model: str | None = None, config=None, max_output_tokens=None):
        if max_output_tokens is None and config is not None:
            max_output_tokens = getattr(config, "max_output_tokens", None)
        return _OpenRouterChat(model=model, max_output_tokens=max_output_tokens)


class _OpenRouterClient:
    def __init__(self):
        self.chats = _OpenRouterChats()


def get_client():
    if _provider() == "openrouter":
        if not is_available():
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return _OpenRouterClient()

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
    if _provider() == "openrouter":
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
    if _provider() == "openrouter":
        content = [{"type": "text", "text": prompt}]
        for image_bytes, mime_type in images:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}"
                },
            })
        return _openrouter_request_with_fallback(
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
    if _provider() == "openrouter":
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
