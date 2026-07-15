from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.cuda_gemm.shapes import selected_shapes


PROVIDER_NAMES = [
    "torch_matmul",
    "cuda_naive",
    "cuda_tiled",
    "cuda_reg_blocked",
    "cuda_vec4",
    "cuda_wmma",
    "cuda_wmma_block_tiled",
]


DEFAULT_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Nsight Compute commands for CUDA GEMM diagnostics.")
    parser.add_argument("--providers", nargs="*", default=["cuda_wmma", "cuda_wmma_block_tiled"], choices=PROVIDER_NAMES)
    parser.add_argument("--shapes", nargs="*", default=["wmma_shape_diagnostic"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape_names = [shape.name for shape in selected_shapes(args.shapes)]
    metric_arg = ",".join(args.metrics)

    print("# Run these on AutoDL RTX 4090. They print to terminal and do not create result files.")
    print("# Copy back the terminal output for attribution.")
    for provider in args.providers:
        for shape_name in shape_names:
            command = (
                "ncu --target-processes all --set full "
                f"--metrics {metric_arg} "
                "python studies/cuda_gemm/benchmark.py "
                f"--dtype float16 --warmup {args.warmup} --repeat {args.repeat} "
                f"--shapes {shape_name} --providers torch_matmul {provider} --no-write"
            )
            print("\n" + command)


if __name__ == "__main__":
    main()
