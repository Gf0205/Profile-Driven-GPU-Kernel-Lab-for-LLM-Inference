from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GemmShape:
    name: str
    m: int
    n: int
    k: int
    model_family: str
    regime: str
    note: str


SHAPES = [
    GemmShape("decode_4096", 1, 4096, 4096, "LLaMA/Qwen-like", "decode", "single-token projection"),
    GemmShape("decode_16_4096", 16, 4096, 4096, "LLaMA/Qwen-like", "decode", "batched decode projection"),
    GemmShape("prefill_128_4096", 128, 4096, 4096, "LLaMA/Qwen-like", "prefill", "medium prefill projection"),
    GemmShape("prefill_512_4096", 512, 4096, 4096, "LLaMA/Qwen-like", "prefill", "large prefill projection"),
    GemmShape("mlp_up_128", 128, 11008, 4096, "LLaMA-7B", "prefill", "MLP up/gate projection"),
    GemmShape("mlp_down_128", 128, 4096, 11008, "LLaMA-7B", "prefill", "MLP down projection"),
    GemmShape("mlp_up_decode", 1, 11008, 4096, "LLaMA-7B", "decode", "single-token MLP up/gate"),
    GemmShape("qwen_mlp_up_128", 128, 18944, 4096, "Qwen-like", "prefill", "wider MLP up/gate"),
]


SHAPE_PRESETS = {
    "wmma_shape_diagnostic": ["qwen_mlp_up_128", "mlp_down_128", "prefill_512_4096"],
}


def selected_shapes(names: list[str] | None = None) -> list[GemmShape]:
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

