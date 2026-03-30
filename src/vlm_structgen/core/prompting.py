from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any

TEMPLATE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def render_prompt_template(template: str, context: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"Prompt template variable {key!r} is missing from context.")
        value = context[key]
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)

    return TEMPLATE_PATTERN.sub(_replace, template)


def build_chat_prompt(
    processor,
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    template_owner = processor if hasattr(processor, "apply_chat_template") else tokenizer
    return template_owner.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@contextmanager
def temporary_padding_side(processor, tokenizer, *, padding_side: str):
    tokenizer_objects: list[Any] = []
    if tokenizer is not None:
        tokenizer_objects.append(tokenizer)
    processor_tokenizer = getattr(processor, "tokenizer", None)
    if processor_tokenizer is not None and processor_tokenizer is not tokenizer:
        tokenizer_objects.append(processor_tokenizer)

    previous: list[tuple[Any, Any]] = []
    try:
        for current in tokenizer_objects:
            previous.append((current, getattr(current, "padding_side", None)))
            current.padding_side = padding_side
        yield
    finally:
        for current, old_value in previous:
            if old_value is not None:
                current.padding_side = old_value
