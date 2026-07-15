# Profile-Driven GPU Kernel Lab for LLM Inference

This repository is an operator-level performance study for real LLM inference
paths. It is intentionally not a broad kernel zoo. Each phase takes one
operator from a realistic inference workload and follows the same loop:

PyTorch baseline -> torch.compile baseline -> Triton/CUDA implementation ->
correctness validation -> realistic shape sweep -> profiling evidence ->
bottleneck attribution -> go/no-go conclusion.

## Project Positioning

The previous nano-vLLM project is system evidence: benchmark audit, scheduler
interference optimization, KV-cache/block metrics, profiling, and go/no-go
discipline. This repository is the main GPU performance signal: CUDA/Triton
operator ownership under realistic LLM inference shapes.

## Study Status

Phase 1, **CUDA GEMM Optimization**, is complete. It followed this ablation:

```text
naive
-> shared-memory tiling
-> register blocking
-> vectorized load
-> Tensor Core / WMMA path
-> block-tiled multi-warp WMMA path
-> cooperative shared-memory staging
-> cuBLAS comparison
-> profiler attribution
-> bottleneck analysis
```

The final cooperative-staging variant removed duplicated per-warp A/B tile
loads and improved three preselected RTX 4090 LLM shapes by 1.32-1.58x over the
block-tiled WMMA kernel. It remains 3.65-7.99x behind cuBLAS, supporting the
final conclusion that high-performance GEMM requires shape-aware kernel
selection and a substantially deeper pipeline. See the
[CUDA GEMM study](studies/cuda_gemm/README.md) for the result table and profiler
attribution.

CUDA GEMM study directory:

```text
studies/cuda_gemm/
  kernels/gemm_kernels.cu  # CUDA kernels for ablation
  gemm_ops.py              # PyTorch extension loader and providers
  shapes.py                # realistic decode/prefill GEMM shapes
  benchmark.py             # no-write cloud benchmark harness
  profiler.py              # no-write profiler table harness
  README.md                # study notes and AutoDL commands
```

Phase 2, **Fused SiLU-Mul**, is also complete. It studies the
activation/multiplication part of SwiGLU MLPs used by LLaMA/Qwen-style models
and establishes the compiler-fusion boundary after the CUDA GEMM main line.
On RTX 4090, manual Triton reaches 1.97x over eager PyTorch for the largest
LLaMA prefill shape, while tiny isolated calls remain launch/dispatch bound.
Profiler evidence shows that `torch.compile` already emits an equivalent fused
Triton kernel within about 2-6% of the manual kernel's GPU execution time. See
the [Fused SiLU-Mul study](studies/fused_silu_mul/README.md).

Fused SiLU-Mul directory:

```text
studies/fused_silu_mul/
  fused_silu_mul.py   # PyTorch, torch.compile, and Triton implementations
  shapes.py           # realistic decode/prefill MLP shapes
  benchmark.py        # repeatable latency/GB/s/correctness sweep
  profiler.py         # optional PyTorch profiler table/trace capture
  README.md           # operator-specific study notes
  results/            # ignored raw CSV/profiler outputs from cloud runs
```

## Why These Two Studies

CUDA GEMM lets the project answer questions that show direct performance
ownership:

- how shared memory changes memory traffic
- why register blocking helps or hurts
- whether vectorized loads matter for realistic shapes
- how far each custom kernel is from cuBLAS
- whether the bottleneck is memory, compute, occupancy, or register pressure

Fused SiLU-Mul complements GEMM with a memory-bound operator and asks when a
manual Triton fusion is useful relative to the strong `torch.compile` baseline.

## Quick Start

Install dependencies on a CUDA machine:

```bash
pip install -r requirements.txt
```

Run the active Fused SiLU-Mul study on the real GPU validation machine. The RTX
4090 is the official validation device for current project results.
Local Windows is used for code editing, docs, git, and result integration; it is
not used for final CUDA/Triton performance conclusions.

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 25 --repeat 100 --shapes silu_official_rtx4090 --no-write
```

Run profiler evidence on AutoDL RTX 4090 without writing trace files:

```bash
python studies/fused_silu_mul/profiler.py --providers pytorch_unfused torch_compile triton --shapes silu_profile_diagnostic --no-write
```

The Fused SiLU-Mul benchmark reports latency, effective GB/s, speedup over
PyTorch unfused, gap versus `torch.compile`, repeat statistics, and correctness
against an FP32 reference.

When running on AutoDL or Colab, copy back the full terminal output, especially
the `BEGIN_BENCHMARK_CSV` / `END_BENCHMARK_CSV` block and profiler tables. The
summary documents are updated locally from that returned output instead of
committing generated result files from the cloud machine.
