"""
OpenCode Zen text client adapter for Cosmic Curious.

Text generation uses a free-model fallback chain:
1. DeepSeek V4.1 Flash Free
2. GLM-5.3-Flash Free
3. Nemotron 3.5 Super VL Free

Visual verification remains on the existing Z.AI vision credential.
"""
from __future__ import annotations

import base64
import os
import json
import subprocess
from types import SimpleNamespace

OPENCODE_BASE_URL = "https://opencode.ai/inference/openai/v1"
TEXT_MODELS = (
    "deepseek-v4.1-flash-free",
    "glm-5.3-flash-free",
    "nemotron-3.5-super-vl-free",
)
TEXT_MODEL = TEXT_MODELS[0]
VISION_BASE_URL = "https://api.z.ai/api/paas/v4/"
VISION_MODEL = "glm-4.6v-flash"


def is_available() -> bool:
    return bool(os.environ.get("OPENCODE_API_KEY"))


def is_vision_available() -> bool:
    return bool(os.environ.get("ZAI_VISION_API_KEY"))


def _openai_client(api_key: str, base_url: str = OPENCODE_BASE_URL):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install dependencies from requirements.txt."
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def _request_with_fallback(kwargs: dict):
    """Run free OpenCode models through the official OpenCode CLI.

    OpenCode intentionally blocks free-tier model requests from external
    API clients. The CLI is the supported route for these free models.
    """
    api_key = os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        raise RuntimeError("OPENCODE_API_KEY is not set.")

    messages = kwargs.get("messages") or []
    prompt_parts = []
    for message in messages:
        if isinstance(message, dict) and message.get("content") is not None:
            prompt_parts.append(str(message["content"]))
    prompt = "\n\n".join(prompt_parts).strip()
    if kwargs.get("response_format", {}).get("type") == "json_object":
        prompt += "\n\nReturn only one valid JSON object. Do not use markdown fences or add commentary."
    if not prompt:
        raise RuntimeError("OpenCode request has no text prompt.")

    last_error = None
    for model in TEXT_MODELS:
        command = [
            "opencode", "run", "--standalone", "--format", "json",
            "--model", f"opencode/{model}",
            "--title", "Cosmic Curious AI generation",
            prompt,
        ]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
                check=False,
            )
            text_parts = []
            event_errors = []
            for line in result.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "text":
                    part = event.get("part") or {}
                    value = part.get("text")
                    if value:
                        text_parts.append(str(value))
                elif event.get("type") == "error":
                    error = event.get("error") or {}
                    data = error.get("data") or {}
                    event_errors.append(
                        f"{data.get('message') or error.get('name') or str(error)} "
                        f"(status={data.get('statusCode', 'unknown')})"
                    )

            content = "\n".join(text_parts).strip()
            if result.returncode != 0 or not content:
                detail = "; ".join(event_errors) or result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(
                    f"OpenCode CLI failed for {model} (exit {result.returncode}): "
                    f"{detail[:1200] or 'no text event returned'}"
                )

            print(f"Text generation succeeded with OpenCode CLI model: {model}")
            choice = SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice])

        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if any(code in message for code in ("401", "403", "unauthorized", "invalid api key", "account is not activated")):
                raise
            print(f"WARNING: OpenCode CLI model {model} failed; trying next fallback: {exc}")

    raise RuntimeError("All OpenCode CLI free models failed: " + str(last_error)) from last_error


class _ChatSession:
    def __init__(self, owner, model: str, config=None):
        self.owner = owner
        self.model = model or TEXT_MODEL
        self.config = config

    def send_message(self, prompt: str):
        max_tokens = getattr(self.config, "max_output_tokens", None)
        response_format = None
        if getattr(self.config, "response_mime_type", None) == "application/json":
            response_format = {"type": "json_object"}
        kwargs = {"messages": [{"role": "user", "content": str(prompt)}]}
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        if response_format:
            kwargs["response_format"] = response_format
        response = _request_with_fallback(kwargs)
        return SimpleNamespace(
            text=(response.choices[0].message.content or "").strip(),
            candidates=[SimpleNamespace(finish_reason=response.choices[0].finish_reason)],
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


class OpenCodeTextClient:
    def __init__(self):
        if not is_available():
            raise RuntimeError("OPENCODE_API_KEY is not set.")
        self.chats = _Chats(self)
        self.models = _Models(self)


def get_client():
    return OpenCodeTextClient()


def generate(prompt: str, max_output_tokens: int | None = None,
             model: str = TEXT_MODEL, json_mode: bool = False) -> str:
    kwargs = {"messages": [{"role": "user", "content": prompt}]}
    if max_output_tokens:
        kwargs["max_tokens"] = int(max_output_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _request_with_fallback(kwargs)
    return (response.choices[0].message.content or "").strip()


def generate_vision(prompt: str, images: list[tuple[bytes, str]], model: str | None = None):
    if not is_vision_available():
        raise RuntimeError("ZAI_VISION_API_KEY is not set.")
    content = [{"type": "text", "text": prompt}]
    for image_bytes, mime_type in images:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        })
    response = _openai_client(
        os.environ["ZAI_VISION_API_KEY"], VISION_BASE_URL
    ).chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1800,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("GLM-4.6V-Flash returned an empty visual audit.")
    return SimpleNamespace(text=text, raw=response)


def generate_response(prompt: str, max_output_tokens: int | None = None,
                      model: str = TEXT_MODEL):
    kwargs = {"messages": [{"role": "user", "content": prompt}]}
    if max_output_tokens:
        kwargs["max_tokens"] = int(max_output_tokens)
    response = _request_with_fallback(kwargs)
    return SimpleNamespace(
        text=(response.choices[0].message.content or "").strip(),
        candidates=[SimpleNamespace(finish_reason=response.choices[0].finish_reason)],
        raw=response,
    )
