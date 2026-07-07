# Phase 1: CUDA GEMM Optimization Study

CUDA GEMM is the first main line of this repository because it is the strongest
screening signal for GPU performance ownership. The goal is not just to
implement `C = A @ B`, but to run a controlled ablation:

```text
naive
-> shared-memory tiling
-> register blocking
-> vectorized load
-> double buffering
-> cuBLAS comparison
-> profiler attribution
-> bottleneck analysis
```

The previous nano-vLLM project provides system context: benchmark audit,
scheduler interference optimization, KV-cache/block metrics, profiling, and
go/no-go discipline. This study turns the GEMM/MLP hotspot into a standalone,
reproducible CUDA performance investigation.

## Current Implementation Status

| Provider | Status | Purpose |
|---|---|---|
| `torch_matmul` | active | cuBLAS-backed practical baseline |
| `cuda_naive` | active | one output element per thread baseline |
| `cuda_tiled` | active | shared-memory tiling ablation |
| `cuda_reg_blocked` | active | per-thread 2x2 output register blocking |
| `cuda_vec4` | active | vectorized contiguous load path for aligned FP16 shapes |
| `cuda_double_buffer` | planned | next ablation after first 3090 correctness/perf pass |

## AutoDL RTX 3090 Setup

The screenshot configuration is suitable for this phase:

- GPU: RTX 3090 24GB, 1 card is enough.
- Driver/CUDA shown by AutoDL: driver branch `570`, CUDA `12.8`.
- Base image shown: `PyTorch / 2.1.0 / 3.10 / ubuntu22.04 / 12.1`.

Driver CUDA being newer than the image CUDA runtime is normal. Use the PyTorch
image first; only switch images if the CUDA extension build fails.

Start the instance, open the terminal, then run:

```bash
git clone https://github.com/Gf0205/Profile-Driven-GPU-Kernel-Lab-for-LLM-Inference.git
cd Profile-Driven-GPU-Kernel-Lab-for-LLM-Inference
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `ninja` is missing for PyTorch CUDA extension builds:

```bash
pip install ninja
```

## Benchmark

Default AutoDL RTX 3090 command. It prints all fields and writes no files:

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --no-write
```

Quick smoke test before the full run:

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 5 --repeat 10 --shapes decode_4096 prefill_128_4096 --no-write
```

Copy back:

- the `ENV ...` line
- every human-readable provider line
- the full `BEGIN_GEMM_CSV` / `END_GEMM_CSV` block
- any extension build error if compilation fails

## Profiler

Profiler tables without writing trace files:

```bash
python studies/cuda_gemm/profile.py --provider all --no-write
```

For a smaller profiler pass:

```bash
python studies/cuda_gemm/profile.py --providers torch_matmul cuda_tiled cuda_reg_blocked --shapes prefill_128_4096 --no-write
```

## Metrics

- `latency_ms`: p50 CUDA-event latency.
- `p20_ms`, `p80_ms`: repeat statistics for stability/noise checks.
- `tflops`: effective `2*M*N*K / latency`.
- `speedup_vs_naive`: provider latency relative to naive CUDA.
- `gap_vs_cublas`: provider latency divided by `torch_matmul` latency.
- `max_diff`, `rel_diff`: correctness against `torch.matmul`.

## Analysis Rules

1. Correctness first. If `max_diff` or `rel_diff` is unreasonable, do not
   discuss performance for that provider/shape.
2. Always compare against cuBLAS-backed `torch_matmul`; beating naive CUDA is
   expected and not enough.
3. Analyze decode-like small `M` separately from prefill-like larger `M`.
4. For small shapes, expect launch overhead and occupancy issues.
5. For large shapes, use TFLOP/s, profiler evidence, memory traffic, occupancy,
   and register pressure to decide the bottleneck.
6. Only continue optimizing a custom CUDA path when the ablation shows a clear
   reason. Falling far behind cuBLAS is still useful if the attribution is
   honest and specific.
