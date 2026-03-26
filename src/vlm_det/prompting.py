from __future__ import annotations

import json
import re
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
