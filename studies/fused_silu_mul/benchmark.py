from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.fused_silu_mul.fused_silu_mul import (  # noqa: E402
    PROVIDERS,
    correctness_reference,
)
from studies.fused_silu_mul.shapes import selected_shapes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Fused SiLU-Mul providers.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shapes", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def cuda_event_ms(fn, warmup: int, repeat: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return statistics.median(times), min(times), max(times)


def effective_gbps(num_elements: int, dtype: torch.dtype, latency_ms: float) -> float:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    traffic_bytes = 3 * num_elements * bytes_per_element
    return traffic_bytes / (latency_ms * 1e-3) / 1e9


def benchmark_provider(provider: str, gate: torch.Tensor, up: torch.Tensor, args: argparse.Namespace):
    fn = PROVIDERS[provider]

    # Compile outside the timed region and force first execution for torch.compile.
    out = fn(gate, up)
    torch.cuda.synchronize()

    ref = correctness_reference(gate, up)
    max_diff = (out.float() - ref).abs().max().item()

    median_ms, min_ms, max_ms = cuda_event_ms(lambda: fn(gate, up), args.warmup, args.repeat)
    gbps = effective_gbps(gate.numel(), gate.dtype, median_ms)
    return {
        "provider": provider,
        "latency_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "gbps": gbps,
        "max_diff": max_diff,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    device_name = torch.cuda.get_device_name()
    rows = []

    for shape in selected_shapes(args.shapes):
        gate = torch.randn(shape.shape, device="cuda", dtype=dtype)
        up = torch.randn(shape.shape, device="cuda", dtype=dtype)

        provider_rows = [
            benchmark_provider(provider, gate, up, args)
            for provider in ["pytorch_unfused", "torch_compile", "triton"]
        ]
        baseline_ms = next(row["latency_ms"] for row in provider_rows if row["provider"] == "pytorch_unfused")
        compile_ms = next(row["latency_ms"] for row in provider_rows if row["provider"] == "torch_compile")

        for row in provider_rows:
            row.update(
                {
                    "device": device_name,
                    "dtype": args.dtype,
                    "shape_name": shape.name,
                    "tokens": shape.tokens,
                    "intermediate": shape.intermediate,
                    "model_family": shape.model_family,
                    "regime": shape.regime,
                    "speedup_vs_pytorch": baseline_ms / row["latency_ms"],
                    "gap_vs_torch_compile": row["latency_ms"] / compile_ms,
                }
            )
            rows.append(row)
            print(
                f"{shape.name:24s} {row['provider']:16s} "
                f"{row['latency_ms']:8.4f} ms {row['gbps']:8.2f} GB/s "
                f"speedup={row['speedup_vs_pytorch']:.2f}x "
                f"compile_gap={row['gap_vs_torch_compile']:.2f}x "
                f"max_diff={row['max_diff']:.3e}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

