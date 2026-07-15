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

from studies.cuda_gemm.gemm_ops import PROVIDERS  # noqa: E402
from studies.cuda_gemm.shapes import selected_shapes  # noqa: E402


DEFAULT_PROVIDERS = [
    "torch_matmul",
    "cuda_tiled",
    "cuda_reg_blocked",
    "cuda_vec4",
    "cuda_wmma",
    "cuda_wmma_block_tiled",
    "cuda_wmma_shared_tiles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CUDA GEMM ablations.")
    parser.add_argument("--dtype", choices=["float16"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
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
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
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


def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    return (2.0 * m * n * k) / (latency_ms * 1e-3) / 1e12


def benchmark_provider(provider: str, a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor, args: argparse.Namespace):
    fn = PROVIDERS[provider]
    out = fn(a, b)
    torch.cuda.synchronize()
    max_diff = (out.float() - ref.float()).abs().max().item()
    rel_diff = max_diff / max(ref.float().abs().max().item(), 1e-12)

    p50_ms, p20_ms, p80_ms, min_ms, max_ms = cuda_event_ms(lambda: fn(a, b), args.warmup, args.repeat)
    return {
        "provider": provider,
        "latency_ms": p50_ms,
        "p20_ms": p20_ms,
        "p80_ms": p80_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "max_diff": max_diff,
        "rel_diff": rel_diff,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    providers = args.providers or DEFAULT_PROVIDERS
    rows = []
    device_name = torch.cuda.get_device_name()
    print(
        f"ENV device={device_name} torch={torch.__version__} "
        f"dtype={args.dtype} warmup={args.warmup} repeat={args.repeat}"
    )

    for shape in selected_shapes(args.shapes):
        a = torch.randn((shape.m, shape.k), device="cuda", dtype=dtype)
        b = torch.randn((shape.k, shape.n), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)
        torch.cuda.synchronize()

        provider_rows = [benchmark_provider(provider, a, b, ref, args) for provider in providers]
        naive_ms = next((row["latency_ms"] for row in provider_rows if row["provider"] == "cuda_naive"), None)
        cublas_ms = next((row["latency_ms"] for row in provider_rows if row["provider"] == "torch_matmul"), None)

        for row in provider_rows:
            row.update(
                {
                    "device": device_name,
                    "dtype": args.dtype,
                    "shape_name": shape.name,
                    "m": shape.m,
                    "n": shape.n,
                    "k": shape.k,
                    "model_family": shape.model_family,
                    "regime": shape.regime,
                    "note": shape.note,
                    "tflops": tflops(shape.m, shape.n, shape.k, row["latency_ms"]),
                    "speedup_vs_naive": (naive_ms / row["latency_ms"]) if naive_ms else "",
                    "gap_vs_cublas": (row["latency_ms"] / cublas_ms) if cublas_ms else "",
                }
            )
            rows.append(row)
            print(
                f"{device_name} {shape.name:18s} {row['provider']:16s} "
                f"p50={row['latency_ms']:8.4f} ms p20={row['p20_ms']:8.4f} ms p80={row['p80_ms']:8.4f} ms "
                f"{row['tflops']:7.2f} TFLOP/s "
                f"naive_speedup={row['speedup_vs_naive']} cublas_gap={row['gap_vs_cublas']} "
                f"max_diff={row['max_diff']:.3e} rel_diff={row['rel_diff']:.3e}"
            )

    if args.no_write:
        print("\nBEGIN_GEMM_CSV")
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        print("END_GEMM_CSV")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.output}")
    else:
        print("\nNo output file requested. Use --no-write for copyable cloud output.")


if __name__ == "__main__":
    main()
