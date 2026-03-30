from __future__ import annotations

from typing import Any

from transformers import StoppingCriteria, StoppingCriteriaList


def normalize_eos_token_ids(eos_token_id: Any) -> int | list[int] | None:
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, (list, tuple, set)):
        normalized = [int(token_id) for token_id in eos_token_id]
        if not normalized:
            return None
        if len(normalized) == 1:
            return normalized[0]
        return normalized
    return int(eos_token_id)


def resolve_eos_token_ids(tokenizer, generation_config=None) -> int | list[int] | None:
    if generation_config is not None:
        generation_eos_id = normalize_eos_token_ids(getattr(generation_config, "eos_token_id", None))
        if generation_eos_id is not None:
            return generation_eos_id
    tokenizer_eos_id = normalize_eos_token_ids(getattr(tokenizer, "eos_token_id", None))
    if tokenizer_eos_id is None:
        return None
    return tokenizer_eos_id


def trim_generated_ids_at_eos(token_ids: Any, eos_token_id: int | list[int] | None) -> list[int]:
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    normalized_eos = normalize_eos_token_ids(eos_token_id)
    if normalized_eos is None:
        return [int(token_id) for token_id in ids]
    eos_ids = {normalized_eos} if isinstance(normalized_eos, int) else set(normalized_eos)
    trimmed: list[int] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in eos_ids:
            break
        trimmed.append(token_id)
    return trimmed


def find_balanced_json_array_end(text: str) -> int | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def has_closed_json_array(text: str) -> bool:
    return find_balanced_json_array_end(text) is not None


class JsonArrayClosureStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_lengths: list[int]) -> None:
        self.tokenizer = tokenizer
        self.prompt_lengths = [int(length) for length in prompt_lengths]

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, ANN003
        batch_size = int(input_ids.shape[0])
        if batch_size != len(self.prompt_lengths):
            return False
        for row_index, prompt_length in enumerate(self.prompt_lengths):
            continuation = input_ids[row_index, prompt_length:]
            continuation_text = self.tokenizer.decode(continuation, skip_special_tokens=False)
            if not has_closed_json_array(continuation_text):
                return False
        return True


def build_json_array_stopping_criteria(tokenizer, prompt_lengths: list[int]) -> StoppingCriteriaList:
    return StoppingCriteriaList([JsonArrayClosureStoppingCriteria(tokenizer, prompt_lengths)])


def build_generate_kwargs(
    tokenizer,
    *,
    generation_config=None,
    num_bins: int,
    prompt_lengths: list[int] | None = None,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
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
    if do_sample:
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
    eos_token_id = resolve_eos_token_ids(tokenizer, generation_config=generation_config)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    return generate_kwargs
