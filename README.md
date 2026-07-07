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

## Current Phase

Phase 1 studies **CUDA GEMM optimization**, because GEMM is the strongest
screening signal for GPU performance engineering. The planned ablation is:

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

Active study directory:

```text
studies/cuda_gemm/
  kernels/gemm_kernels.cu  # CUDA kernels for ablation
  gemm_ops.py              # PyTorch extension loader and providers
  shapes.py                # realistic decode/prefill GEMM shapes
  benchmark.py             # no-write cloud benchmark harness
  profile.py               # no-write profiler table harness
  README.md                # study notes and AutoDL commands
```

Phase 2 studies **Fused SiLU-Mul**, the activation/multiplication part of
SwiGLU MLPs used by LLaMA/Qwen-style models. It is kept as the compiler-fusion
boundary study after the CUDA GEMM main line.

Phase 3 candidate: **W4A16 GEMM feasibility / quantized inference study** after
auditing packing, scale layout, and dequant correctness.

Fused SiLU-Mul directory:

```text
studies/fused_silu_mul/
  fused_silu_mul.py   # PyTorch, torch.compile, and Triton implementations
  shapes.py           # realistic decode/prefill MLP shapes
  benchmark.py        # repeatable latency/GB/s/correctness sweep
  profile.py          # optional PyTorch profiler table/trace capture
  README.md           # operator-specific study notes
  results/            # CSV/profiler outputs from T4/3090/A100 runs
```

## Why CUDA GEMM First

CUDA GEMM lets the project answer questions that show direct performance
ownership:

- how shared memory changes memory traffic
- why register blocking helps or hurts
- whether vectorized loads matter for realistic shapes
- whether double buffering is effective
- how far each custom kernel is from cuBLAS
- whether the bottleneck is memory, compute, occupancy, or register pressure

## Quick Start

Install dependencies on a CUDA machine:

```bash
pip install -r requirements.txt
```

Run the phase-1 CUDA GEMM benchmark on the real GPU validation machine. Local Windows is
used for code editing, docs, git, and result integration; it is not used for
final CUDA/Triton performance conclusions.

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --no-write
```

Run profiler evidence on AutoDL RTX 3090 without writing trace files:

```bash
python studies/cuda_gemm/profile.py --provider all --no-write
```

The CUDA GEMM benchmark reports latency, TFLOP/s, speedup vs naive CUDA, gap vs
cuBLAS-backed `torch.matmul`, and correctness against `torch.matmul`.

When running on AutoDL or Colab, copy back the full terminal output, especially
the `BEGIN_GEMM_CSV` / `END_GEMM_CSV` block and profiler tables. The
summary documents are updated locally from that returned output instead of
committing generated result files from the cloud machine.
