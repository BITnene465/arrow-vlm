from __future__ import annotations

from typing import Any

from vlm_det.protocol.tokens import (
    ARROW_BEGIN_TOKEN,
    ARROW_END_TOKEN,
    ARROWS_BEGIN_TOKEN,
    ARROWS_END_TOKEN,
    BOX_BEGIN_TOKEN,
    BOX_END_TOKEN,
    OCCLUDED_TOKEN,
    POINT_BEGIN_TOKEN,
    POINT_END_TOKEN,
    POINTS_BEGIN_TOKEN,
    POINTS_END_TOKEN,
    VISIBLE_TOKEN,
    x_token,
    y_token,
)


class ArrowProtocolConstraint:
    def __init__(self, tokenizer, num_bins: int, prompt_lengths: list[int]) -> None:
        self.tokenizer = tokenizer
        self.num_bins = num_bins
        self.prompt_lengths = prompt_lengths

        self.arrows_begin_id = self._token_id(ARROWS_BEGIN_TOKEN)
        self.arrows_end_id = self._token_id(ARROWS_END_TOKEN)
        self.arrow_begin_id = self._token_id(ARROW_BEGIN_TOKEN)
        self.arrow_end_id = self._token_id(ARROW_END_TOKEN)
        self.box_begin_id = self._token_id(BOX_BEGIN_TOKEN)
        self.box_end_id = self._token_id(BOX_END_TOKEN)
        self.points_begin_id = self._token_id(POINTS_BEGIN_TOKEN)
        self.points_end_id = self._token_id(POINTS_END_TOKEN)
        self.point_begin_id = self._token_id(POINT_BEGIN_TOKEN)
        self.point_end_id = self._token_id(POINT_END_TOKEN)
        self.visible_id = self._token_id(VISIBLE_TOKEN)
        self.occluded_id = self._token_id(OCCLUDED_TOKEN)
        self.x_ids = [self._token_id(x_token(index)) for index in range(num_bins)]
        self.y_ids = [self._token_id(y_token(index)) for index in range(num_bins)]
        self.visibility_ids = [self.visible_id, self.occluded_id]
        self.protocol_token_ids = {
            self.arrows_begin_id,
            self.arrows_end_id,
            self.arrow_begin_id,
            self.arrow_end_id,
            self.box_begin_id,
            self.box_end_id,
            self.points_begin_id,
            self.points_end_id,
            self.point_begin_id,
            self.point_end_id,
            self.visible_id,
            self.occluded_id,
            *self.x_ids,
            *self.y_ids,
        }

    def __call__(self, batch_id: int, input_ids) -> list[int]:
        prompt_length = self.prompt_lengths[batch_id]
        generated = input_ids[prompt_length:].tolist()
        state = self._advance_state(generated)
        return self._allowed_tokens(state)

    def _token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Token {token} not found in tokenizer vocabulary.")
        return int(token_id)

    def _advance_state(self, generated: list[int]) -> tuple[str, int]:
        state = "expect_arrows_begin"
        points_in_current_arrow = 0
        for token_id in generated:
            if state == "expect_arrows_begin":
                if token_id != self.arrows_begin_id:
                    return ("invalid", 0)
                state = "expect_arrow_or_end"
            elif state == "expect_arrow_or_end":
                if token_id == self.arrow_begin_id:
                    state = "expect_box_begin"
                    points_in_current_arrow = 0
                elif token_id == self.arrows_end_id:
                    state = "done"
                else:
                    return ("invalid", 0)
            elif state == "expect_box_begin":
                if token_id != self.box_begin_id:
                    return ("invalid", 0)
                state = "expect_box_x1"
            elif state == "expect_box_x1":
                if token_id not in self.x_ids:
                    return ("invalid", 0)
                state = "expect_box_y1"
            elif state == "expect_box_y1":
                if token_id not in self.y_ids:
                    return ("invalid", 0)
                state = "expect_box_x2"
            elif state == "expect_box_x2":
                if token_id not in self.x_ids:
                    return ("invalid", 0)
                state = "expect_box_y2"
            elif state == "expect_box_y2":
                if token_id not in self.y_ids:
                    return ("invalid", 0)
                state = "expect_box_end"
            elif state == "expect_box_end":
                if token_id != self.box_end_id:
                    return ("invalid", 0)
                state = "expect_points_begin"
            elif state == "expect_points_begin":
                if token_id != self.points_begin_id:
                    return ("invalid", 0)
                state = "expect_point_begin_required"
            elif state == "expect_point_begin_required":
                if token_id != self.point_begin_id:
                    return ("invalid", 0)
                state = "expect_point_x"
            elif state == "expect_point_begin_or_end":
                if token_id == self.point_begin_id:
                    state = "expect_point_x"
                elif token_id == self.points_end_id:
                    state = "expect_arrow_end"
                else:
                    return ("invalid", 0)
            elif state == "expect_point_x":
                if token_id not in self.x_ids:
                    return ("invalid", 0)
                state = "expect_point_y"
            elif state == "expect_point_y":
                if token_id not in self.y_ids:
                    return ("invalid", 0)
                state = "expect_visibility"
            elif state == "expect_visibility":
                if token_id not in self.visibility_ids:
                    return ("invalid", 0)
                state = "expect_point_end"
            elif state == "expect_point_end":
                if token_id != self.point_end_id:
                    return ("invalid", 0)
                points_in_current_arrow += 1
                state = "expect_point_begin_or_end" if points_in_current_arrow >= 2 else "expect_point_begin_required"
            elif state == "expect_arrow_end":
                if token_id != self.arrow_end_id:
                    return ("invalid", 0)
                state = "expect_arrow_or_end"
            elif state == "done":
                return ("done", points_in_current_arrow)
            else:
                return ("invalid", 0)
        return (state, points_in_current_arrow)

    def _allowed_tokens(self, state_info: tuple[str, int]) -> list[int]:
        state, _ = state_info
        if state == "expect_arrows_begin":
            return [self.arrows_begin_id]
        if state == "expect_arrow_or_end":
            return [self.arrow_begin_id, self.arrows_end_id]
        if state == "expect_box_begin":
            return [self.box_begin_id]
        if state == "expect_box_x1" or state == "expect_box_x2" or state == "expect_point_x":
            return self.x_ids
        if state == "expect_box_y1" or state == "expect_box_y2" or state == "expect_point_y":
            return self.y_ids
        if state == "expect_box_end":
            return [self.box_end_id]
        if state == "expect_points_begin":
            return [self.points_begin_id]
        if state == "expect_point_begin_required":
            return [self.point_begin_id]
        if state == "expect_point_begin_or_end":
            return [self.point_begin_id, self.points_end_id]
        if state == "expect_visibility":
            return self.visibility_ids
        if state == "expect_point_end":
            return [self.point_end_id]
        if state == "expect_arrow_end":
            return [self.arrow_end_id]
        if state == "done":
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                return [int(eos_id)]
            return [self.arrows_end_id]
        # Fallback: if parsing somehow desynchronized, keep generation inside protocol vocabulary.
        return sorted(self.protocol_token_ids)


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
    num_bins: int,
    prompt_lengths: list[int] | None = None,
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
    if prompt_lengths is not None:
        constraint = ArrowProtocolConstraint(
            tokenizer=tokenizer,
            num_bins=num_bins,
            prompt_lengths=prompt_lengths,
        )
        generate_kwargs["prefix_allowed_tokens_fn"] = constraint
    return generate_kwargs
