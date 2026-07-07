from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.cuda_gemm.gemm_ops import PROVIDERS  # noqa: E402
from studies.cuda_gemm.shapes import selected_shapes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CUDA GEMM ablations.")
    parser.add_argument("--provider", choices=["all", *PROVIDERS.keys()], default="all")
    parser.add_argument("--providers", nargs="*", default=None, choices=list(PROVIDERS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--shapes", nargs="*", default=["decode_4096", "prefill_128_4096"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def run_profile(provider: str, shape_name: str, a: torch.Tensor, b: torch.Tensor, args: argparse.Namespace) -> None:
    fn = PROVIDERS[provider]
    for _ in range(args.warmup):
        fn(a, b)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            with record_function(f"{provider}_{shape_name}"):
                fn(a, b)
    torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(f"\nBEGIN_GEMM_PROFILER_TABLE provider={provider} shape={shape_name}")
    print(table)
    print(f"END_GEMM_PROFILER_TABLE provider={provider} shape={shape_name}")

    if not args.no_write:
        if args.output_dir is None:
            raise ValueError("--output-dir is required unless --no-write is set.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.output_dir / f"{shape_name}_{provider}.json"
        table_path = args.output_dir / f"{shape_name}_{provider}.txt"
        prof.export_chrome_trace(str(trace_path))
        table_path.write_text(table, encoding="utf-8")
        print(f"Wrote {trace_path}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for profiling.")

    providers = args.providers or (list(PROVIDERS) if args.provider == "all" else [args.provider])
    print(
        f"ENV device={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"warmup={args.warmup} steps={args.steps}"
    )

    for shape in selected_shapes(args.shapes):
        a = torch.randn((shape.m, shape.k), device="cuda", dtype=torch.float16)
        b = torch.randn((shape.k, shape.n), device="cuda", dtype=torch.float16)
        for provider in providers:
            run_profile(provider, shape.name, a, b, args)


if __name__ == "__main__":
    main()
