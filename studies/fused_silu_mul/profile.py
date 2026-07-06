from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.fused_silu_mul.fused_silu_mul import PROVIDERS  # noqa: E402
from studies.fused_silu_mul.shapes import selected_shapes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Fused SiLU-Mul providers.")
    parser.add_argument("--provider", choices=["all", *PROVIDERS.keys()], default="all")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--shapes",
        nargs="*",
        default=["llama7b_decode_b1", "llama7b_prefill_1024"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_profile(provider: str, shape_name: str, gate: torch.Tensor, up: torch.Tensor, args: argparse.Namespace) -> None:
    fn = PROVIDERS[provider]
    for _ in range(args.warmup):
        fn(gate, up)
    torch.cuda.synchronize()

    trace_path = args.output_dir / f"{shape_name}_{provider}.json"
    table_path = args.output_dir / f"{shape_name}_{provider}.txt"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            with record_function(f"{provider}_{shape_name}"):
                fn(gate, up)
    torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    table_path.write_text(table, encoding="utf-8")
    print(f"Wrote {trace_path}")
    print(table)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for profiling.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    providers = list(PROVIDERS) if args.provider == "all" else [args.provider]

    for shape in selected_shapes(args.shapes):
        gate = torch.randn(shape.shape, device="cuda", dtype=dtype)
        up = torch.randn(shape.shape, device="cuda", dtype=dtype)
        for provider in providers:
            run_profile(provider, shape.name, gate, up, args)


if __name__ == "__main__":
    main()

