from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import triton

from studies.fused_silu_mul.fused_silu_mul import (  # noqa: E402
    PROVIDERS,
    correctness_reference,
)
from studies.fused_silu_mul.shapes import selected_shapes  # noqa: E402


DEFAULT_PROVIDERS = ["pytorch_unfused", "torch_compile", "triton"]
TOLERANCES = {
    "float16": (2e-2, 2e-3),
    "bfloat16": (1e-1, 1e-2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Fused SiLU-Mul providers.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--amortized-inner",
        type=int,
        default=100,
        help="Calls enclosed by one CUDA-event pair for amortized timing.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shapes", nargs="*", default=None)
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        choices=list(PROVIDERS),
        help=f"Providers to run. Default: {' '.join(DEFAULT_PROVIDERS)}",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print results only. Use this on cloud benchmark machines to avoid local result files.",
    )
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def cuda_event_ms(fn, warmup: int, repeat: int) -> tuple[float, float, float, float, float]:
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

    sorted_times = sorted(times)
    return (
        statistics.median(times),
        percentile(sorted_times, 0.20),
        percentile(sorted_times, 0.80),
        min(times),
        max(times),
    )


def amortized_cuda_event_ms(
    fn,
    warmup: int,
    repeat: int,
    inner: int,
) -> tuple[float, float, float, float, float]:
    if inner < 1:
        raise ValueError("--amortized-inner must be at least 1.")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / inner)

    sorted_times = sorted(times)
    return (
        statistics.median(times),
        percentile(sorted_times, 0.20),
        percentile(sorted_times, 0.80),
        min(times),
        max(times),
    )


def effective_gbps(num_elements: int, dtype: torch.dtype, latency_ms: float) -> float:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    traffic_bytes = 3 * num_elements * bytes_per_element
    return traffic_bytes / (latency_ms * 1e-3) / 1e9


def benchmark_provider(
    provider: str,
    gate: torch.Tensor,
    up: torch.Tensor,
    ref: torch.Tensor,
    args: argparse.Namespace,
):
    fn = PROVIDERS[provider]

    # Compile outside the timed region and force first execution for torch.compile.
    out = fn(gate, up)
    torch.cuda.synchronize()

    max_diff = (out.float() - ref).abs().max().item()
    ref_abs_max = ref.abs().max().item()
    rel_diff = max_diff / max(ref_abs_max, 1e-12)
    atol, rtol = TOLERANCES[args.dtype]
    correct = torch.allclose(out.float(), ref, atol=atol, rtol=rtol)

    median_ms, p20_ms, p80_ms, min_ms, max_ms = cuda_event_ms(lambda: fn(gate, up), args.warmup, args.repeat)
    amortized_ms, amortized_p20_ms, amortized_p80_ms, amortized_min_ms, amortized_max_ms = (
        amortized_cuda_event_ms(
            lambda: fn(gate, up),
            args.warmup,
            args.repeat,
            args.amortized_inner,
        )
    )
    gbps = effective_gbps(gate.numel(), gate.dtype, median_ms)
    amortized_gbps = effective_gbps(gate.numel(), gate.dtype, amortized_ms)
    return {
        "provider": provider,
        "latency_ms": median_ms,
        "p20_ms": p20_ms,
        "p80_ms": p80_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "gbps": gbps,
        "amortized_ms": amortized_ms,
        "amortized_p20_ms": amortized_p20_ms,
        "amortized_p80_ms": amortized_p80_ms,
        "amortized_min_ms": amortized_min_ms,
        "amortized_max_ms": amortized_max_ms,
        "amortized_gbps": amortized_gbps,
        "max_diff": max_diff,
        "rel_diff": rel_diff,
        "correct": correct,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype = getattr(torch, args.dtype)
    providers = args.providers or DEFAULT_PROVIDERS
    atol, rtol = TOLERANCES[args.dtype]
    torch.manual_seed(args.seed)
    device_name = torch.cuda.get_device_name()
    rows = []
    print(
        f"ENV device={device_name} torch={torch.__version__} "
        f"triton={triton.__version__} cuda={torch.version.cuda} "
        f"capability={torch.cuda.get_device_capability()} dtype={args.dtype} "
        f"atol={atol} rtol={rtol} "
        f"warmup={args.warmup} repeat={args.repeat} "
        f"amortized_inner={args.amortized_inner}"
    )

    for shape in selected_shapes(args.shapes):
        gate = torch.randn(shape.shape, device="cuda", dtype=dtype)
        up = torch.randn(shape.shape, device="cuda", dtype=dtype)
        ref = correctness_reference(gate, up)

        provider_rows = [
            benchmark_provider(provider, gate, up, ref, args)
            for provider in providers
        ]
        baseline_ms = next(
            (row["latency_ms"] for row in provider_rows if row["provider"] == "pytorch_unfused"),
            None,
        )
        compile_ms = next(
            (row["latency_ms"] for row in provider_rows if row["provider"] == "torch_compile"),
            None,
        )
        baseline_amortized_ms = next(
            (row["amortized_ms"] for row in provider_rows if row["provider"] == "pytorch_unfused"),
            None,
        )
        compile_amortized_ms = next(
            (row["amortized_ms"] for row in provider_rows if row["provider"] == "torch_compile"),
            None,
        )

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
                    "speedup_vs_pytorch": (baseline_ms / row["latency_ms"]) if baseline_ms else "",
                    "gap_vs_torch_compile": (row["latency_ms"] / compile_ms) if compile_ms else "",
                    "amortized_speedup_vs_pytorch": (
                        baseline_amortized_ms / row["amortized_ms"]
                        if baseline_amortized_ms
                        else ""
                    ),
                    "amortized_gap_vs_torch_compile": (
                        row["amortized_ms"] / compile_amortized_ms
                        if compile_amortized_ms
                        else ""
                    ),
                }
            )
            rows.append(row)
            speedup_display = (
                "" if row["speedup_vs_pytorch"] == "" else f"{row['speedup_vs_pytorch']:.2f}x"
            )
            compile_gap_display = (
                "" if row["gap_vs_torch_compile"] == "" else f"{row['gap_vs_torch_compile']:.2f}x"
            )
            amortized_speedup_display = (
                ""
                if row["amortized_speedup_vs_pytorch"] == ""
                else f"{row['amortized_speedup_vs_pytorch']:.2f}x"
            )
            amortized_compile_gap_display = (
                ""
                if row["amortized_gap_vs_torch_compile"] == ""
                else f"{row['amortized_gap_vs_torch_compile']:.2f}x"
            )
            print(
                f"{device_name} {shape.name:24s} {row['provider']:16s} "
                f"isolated_p50={row['latency_ms']:8.4f} ms "
                f"isolated_p20={row['p20_ms']:8.4f} ms isolated_p80={row['p80_ms']:8.4f} ms "
                f"amortized_p50={row['amortized_ms']:8.4f} ms "
                f"amortized_p20={row['amortized_p20_ms']:8.4f} ms "
                f"amortized_p80={row['amortized_p80_ms']:8.4f} ms "
                f"isolated_gbps={row['gbps']:8.2f} amortized_gbps={row['amortized_gbps']:8.2f} "
                f"isolated_speedup={speedup_display} isolated_compile_gap={compile_gap_display} "
                f"amortized_speedup={amortized_speedup_display} "
                f"amortized_compile_gap={amortized_compile_gap_display} "
                f"max_diff={row['max_diff']:.3e} rel_diff={row['rel_diff']:.3e} "
                f"correct={row['correct']}"
            )

    if args.no_write:
        print("\nBEGIN_BENCHMARK_CSV")
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        print("END_BENCHMARK_CSV")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")
    else:
        print("\nNo output file requested. Re-run with --output PATH to save CSV, or --no-write for copyable CSV output.")


if __name__ == "__main__":
    main()
