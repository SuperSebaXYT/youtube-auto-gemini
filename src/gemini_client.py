"""
Shared Gemini text and vision client.

Cosmic Curious uses Gemini for every AI task: topic generation, scripts,
scientific review, editorial agents, and visual verification.
"""

from __future__ import annotations

import os

GEMINI_MODEL = "gemini-3.5-flash-lite"


def _provider() -> str:
    # Gemini is the only supported provider in this pipeline.
    return "gemini"


def is_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai
    return genai.Client(api_key=api_key)


def generate(
    prompt: str,
    max_output_tokens: int | None = None,
    model: str = GEMINI_MODEL,
    json_mode: bool = False,
) -> str:
    """Generate text with Gemini."""
    from google.genai import types

    client = get_client()
    if max_output_tokens:
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        chat = client.chats.create(model=model, config=config)
    else:
        chat = client.chats.create(model=model)

    response = chat.send_message(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def generate_vision(
    prompt: str,
    images: list[tuple[bytes, str]],
    model: str | None = None,
):
    """Inspect image bytes with Gemini vision."""
    from google.genai import types

    client = get_client()
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
    """Generate text and return the complete Gemini response."""
    from google.genai import types

    client = get_client()
    if max_output_tokens:
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens
        )
        chat = client.chats.create(model=model, config=config)
    else:
        chat = client.chats.create(model=model)

    return chat.send_message(prompt)
