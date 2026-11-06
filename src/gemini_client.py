"""
Z.AI client adapter for Cosmic Curious.

Text generation uses GLM-4.7-Flash with ZAI_TEXT_API_KEY.
Visual verification uses GLM-4.6V-Flash with ZAI_VISION_API_KEY.
The two credentials are deliberately isolated.
"""
from __future__ import annotations

import base64
import os
from types import SimpleNamespace

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4/"
TEXT_MODEL = "glm-4.7-flash"
VISION_MODEL = "glm-4.6v-flash"


def is_available() -> bool:
    return bool(os.environ.get("ZAI_TEXT_API_KEY"))


def is_vision_available() -> bool:
    return bool(os.environ.get("ZAI_VISION_API_KEY"))


def _openai_client(api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required for the Z.AI API. "
            "Install dependencies from requirements.txt."
        ) from exc
    return OpenAI(api_key=api_key, base_url=ZAI_BASE_URL, timeout=120.0)


class _ChatSession:
    def __init__(self, owner, model: str, config=None):
        self.owner = owner
        self.model = TEXT_MODEL
        self.config = config

    def send_message(self, prompt: str):
        max_tokens = getattr(self.config, "max_output_tokens", None)
        response_format = None
        mime_type = getattr(self.config, "response_mime_type", None)
        if mime_type == "application/json":
            response_format = {"type": "json_object"}

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
        }
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        if response_format:
            kwargs["response_format"] = response_format

        response = self.owner._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = (message.content or "").strip()
        return SimpleNamespace(
            text=text,
            candidates=[
                SimpleNamespace(finish_reason=response.choices[0].finish_reason)
            ],
            raw=response,
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
            return generate_vision("\n".join(prompt_parts), images)
        return generate_response("\n".join(prompt_parts))


class ZAITextClient:
    def __init__(self):
        api_key = os.environ.get("ZAI_TEXT_API_KEY")
        if not api_key:
            raise RuntimeError("ZAI_TEXT_API_KEY is not set.")
        self._client = _openai_client(api_key)
        self.chats = _Chats(self)
        self.models = _Models(self)


def get_client():
    """Return a text client compatible with the existing pipeline call sites."""
    return ZAITextClient()


def generate(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = TEXT_MODEL,
    json_mode: bool = False,
) -> str:
    if not is_available():
        raise RuntimeError("ZAI_TEXT_API_KEY is not set.")
    kwargs = {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_output_tokens:
        kwargs["max_tokens"] = int(max_output_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _openai_client(os.environ["ZAI_TEXT_API_KEY"]).chat.completions.create(**kwargs)
    result = (response.choices[0].message.content or "").strip()
    if not result:
        raise RuntimeError("GLM-4.7-Flash returned an empty response.")
    return result


def generate_vision(
    prompt: str,
    images: list[tuple[bytes, str]],
    model: str | None = None,
):
    """Send image frames only to GLM-4.6V-Flash using the vision secret."""
    if not is_vision_available():
        raise RuntimeError("ZAI_VISION_API_KEY is not set.")

    content = [{"type": "text", "text": prompt}]
    for image_bytes, mime_type in images:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        })

    response = _openai_client(os.environ["ZAI_VISION_API_KEY"]).chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1800,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("GLM-4.6V-Flash returned an empty visual audit.")
    return SimpleNamespace(text=text, raw=response)


def generate_response(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = TEXT_MODEL,
):
    if not is_available():
        raise RuntimeError("ZAI_TEXT_API_KEY is not set.")
    kwargs = {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_output_tokens:
        kwargs["max_tokens"] = int(max_output_tokens)
    response = _openai_client(os.environ["ZAI_TEXT_API_KEY"]).chat.completions.create(**kwargs)
    return SimpleNamespace(
        text=(response.choices[0].message.content or "").strip(),
        candidates=[
            SimpleNamespace(finish_reason=response.choices[0].finish_reason)
        ],
        raw=response,
    )
