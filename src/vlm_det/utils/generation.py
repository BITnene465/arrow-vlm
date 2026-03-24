from __future__ import annotations

from typing import Any

def resolve_eos_token_ids(tokenizer) -> int | list[int] | None:
    tokenizer_eos_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos_id is None:
        return None
    return int(tokenizer_eos_id)


def build_generate_kwargs(
    tokenizer,
    *,
    num_bins: int,
    prompt_lengths: list[int] | None = None,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    use_cache: bool,
) -> dict[str, Any]:
    del num_bins, prompt_lengths
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
