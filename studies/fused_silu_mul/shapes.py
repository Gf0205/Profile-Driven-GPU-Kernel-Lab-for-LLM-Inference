from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPShape:
    name: str
    tokens: int
    intermediate: int
    model_family: str
    regime: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tokens, self.intermediate)

    @property
    def elements(self) -> int:
        return self.tokens * self.intermediate


SHAPES = [
    MLPShape("llama7b_decode_b1", 1, 11008, "LLaMA-7B", "decode"),
    MLPShape("llama7b_decode_b16", 16, 11008, "LLaMA-7B", "decode"),
    MLPShape("llama7b_decode_b32", 32, 11008, "LLaMA-7B", "decode"),
    MLPShape("llama7b_prefill_128", 128, 11008, "LLaMA-7B", "prefill"),
    MLPShape("llama7b_prefill_1024", 1024, 11008, "LLaMA-7B", "prefill"),
    MLPShape("llama13b_decode_b1", 1, 13824, "LLaMA-13B", "decode"),
    MLPShape("llama13b_prefill_512", 512, 13824, "LLaMA-13B", "prefill"),
    MLPShape("qwen_like_decode_b1", 1, 18944, "Qwen-like", "decode"),
    MLPShape("qwen_like_decode_b16", 16, 18944, "Qwen-like", "decode"),
    MLPShape("qwen_like_prefill_512", 512, 18944, "Qwen-like", "prefill"),
    MLPShape("wide_mlp_decode_b1", 1, 28672, "wide-MLP", "decode"),
    MLPShape("wide_mlp_prefill_256", 256, 28672, "wide-MLP", "prefill"),
]


SHAPE_PRESETS = {
    "silu_official_rtx4090": [
        "llama7b_decode_b1",
        "llama7b_decode_b16",
        "llama7b_prefill_128",
        "llama7b_prefill_1024",
        "qwen_like_decode_b1",
        "qwen_like_prefill_512",
    ],
    "silu_profile_diagnostic": [
        "llama7b_decode_b1",
        "llama7b_prefill_1024",
        "qwen_like_prefill_512",
    ],
}


def selected_shapes(names: list[str] | None = None) -> list[MLPShape]:
    if not names:
        return SHAPES

    expanded_names = []
    for name in names:
        expanded_names.extend(SHAPE_PRESETS.get(name, [name]))
    by_name = {shape.name: shape for shape in SHAPES}
    missing = sorted(set(expanded_names) - set(by_name))
    if missing:
        known_presets = ", ".join(sorted(SHAPE_PRESETS))
        raise ValueError(f"Unknown shape names: {missing}. Known presets: {known_presets}")
    return [by_name[name] for name in expanded_names]
