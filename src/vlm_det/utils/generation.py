from __future__ import annotations

from typing import Any

from vlm_det.protocol.tokens import ARROWS_END_TOKEN


def resolve_eos_token_ids(tokenizer) -> int | list[int] | None:
    eos_token_ids: list[int] = []
    tokenizer_eos_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos_id is not None:
        eos_token_ids.append(int(tokenizer_eos_id))
    arrows_end_id = tokenizer.convert_tokens_to_ids(ARROWS_END_TOKEN)
    if arrows_end_id is not None and arrows_end_id >= 0:
        eos_token_ids.append(int(arrows_end_id))
    eos_token_ids = list(dict.fromkeys(eos_token_ids))
    if not eos_token_ids:
        return None
    if len(eos_token_ids) == 1:
        return eos_token_ids[0]
    return eos_token_ids


def build_generate_kwargs(
    tokenizer,
    *,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    use_cache: bool,
) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "use_cache": use_cache,
        "pad_token_id": tokenizer.pad_token_id,
    }
    eos_token_id = resolve_eos_token_ids(tokenizer)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    return generate_kwargs
