"""
Unified text and vision model gateway for Cosmic Curious.

Shared fallback order:
1. DeepSeek V4.1 Flash Free (OpenCode) - primary
2. GLM-5.3 Flash Free (OpenCode) - fallback 1
3. Nemotron 3.5 Super VL Free (OpenCode) - fallback 2
"""
from __future__ import annotations

import os
import json
import re
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

MODEL_CHAIN = (
    ("opencode", "deepseek-v4.1-flash-free"),
    ("opencode", "glm-5.3-flash-free"),
    ("opencode", "nemotron-3.5-super-vl-free"),
)
TEXT_MODEL = "deepseek-v4.1-flash-free"


def is_available() -> bool:
    return bool(os.environ.get("OPENCODE_API_KEY"))


def is_vision_available() -> bool:
    """Compatibility check for callers; vision uses the same OpenCode model chain."""
    return is_available()


def _openai_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install dependencies from requirements.txt."
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


def _prompt_from_kwargs(kwargs: dict) -> str:
    messages = kwargs.get("messages") or []
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    prompt = "\n\n".join(parts).strip()
    if kwargs.get("response_format", {}).get("type") == "json_object":
        prompt += "\n\nReturn only one valid JSON object. Do not use markdown fences or add commentary."
    if not prompt:
        raise RuntimeError("AI request has no text prompt.")
    prompt = (
        "SYSTEM CONSTRAINT: Do not call or request tools, web search, browser, shell, "
        "or external functions. Answer directly using the supplied prompt and context only.\\n\\n"
        + prompt
    )
    return prompt


def _run_opencode_model(model: str, prompt: str, image_paths: list[str] | None = None):
    command = [
        "opencode", "run", "--standalone", "--format", "json",
        "--model", f"opencode/{model}",
        "--title", "Cosmic Curious unified AI gateway",
    ]
    for image_path in image_paths or []:
        command.extend(["--file", image_path])
    command.append(prompt)
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace", timeout=240, check=False,
    )
    text_parts, event_errors = [], []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "text":
            value = (event.get("part") or {}).get("text")
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
    return content




def _validate_json_output(content: str) -> None:
    """Reject malformed or truncated JSON so the next fallback model is tried."""
    candidate = content.strip()
    candidate = re.sub(r"^\\s*```(?:json)?\\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\\s*```\\s*$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a complete JSON object.")
    json.loads(candidate[start:end + 1])


def _request_with_fallback(kwargs: dict, images: list[tuple[bytes, str]] | None = None):
    """Run text and image requests through the same ordered model chain."""
    if not os.environ.get("OPENCODE_API_KEY"):
        raise RuntimeError("OPENCODE_API_KEY is not set.")
    prompt = _prompt_from_kwargs(kwargs)
    last_error = None
    with tempfile.TemporaryDirectory(prefix="cosmic_ai_") as temp_dir:
        image_paths = []
        for index, (image_bytes, mime_type) in enumerate(images or []):
            extension = ".png" if mime_type == "image/png" else ".jpg"
            image_path = str(Path(temp_dir) / f"frame_{index + 1}{extension}")
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            image_paths.append(image_path)

        for provider, model in MODEL_CHAIN:
            try:
                content = _run_opencode_model(model, prompt, image_paths)
                if kwargs.get("response_format", {}).get("type") == "json_object":
                    _validate_json_output(content)
                print(f"AI generation succeeded with model: {model} ({provider})")
                choice = SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason="stop"
                )
                return SimpleNamespace(choices=[choice])
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if any(code in message for code in (
                    "401", "403", "unauthorized", "invalid api key",
                    "account is not activated",
                )):
                    raise
                print(
                    f"WARNING: AI model {model} ({provider}) failed; "
                    f"trying next fallback: {exc}"
                )
    raise RuntimeError("All unified AI fallback models failed: " + str(last_error)) from last_error


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
        prompt_lower = str(prompt).lower()
        if "return only valid json" in prompt_lower or "return only one valid json object" in prompt_lower:
            kwargs["response_format"] = {"type": "json_object"}
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
    if not images:
        raise RuntimeError("Visual verification requires at least one image.")
    response = _request_with_fallback(
        {"messages": [{"role": "user", "content": prompt}]}, images=images
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("All vision fallback models returned empty output.")
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
