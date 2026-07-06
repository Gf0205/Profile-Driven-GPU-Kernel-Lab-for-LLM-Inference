# Profile-Driven GPU Kernel Lab for LLM Inference

This repository is an operator-level performance study for real LLM inference
paths. It is intentionally not a broad kernel zoo. Each phase takes one
operator from a realistic inference workload and follows the same loop:

PyTorch baseline -> torch.compile baseline -> Triton/CUDA implementation ->
correctness validation -> realistic shape sweep -> profiling evidence ->
bottleneck attribution -> go/no-go conclusion.

## Current Phase

Phase 1 studies **Fused SiLU-Mul**, the activation/multiplication part of
SwiGLU MLPs used by LLaMA/Qwen-style models.

Study directory:

```text
studies/fused_silu_mul/
  fused_silu_mul.py   # PyTorch, torch.compile, and Triton implementations
  shapes.py           # realistic decode/prefill MLP shapes
  benchmark.py        # repeatable latency/GB/s/correctness sweep
  profile.py          # optional PyTorch profiler table/trace capture
  README.md           # operator-specific study notes
  results/            # CSV/profiler outputs from T4/3090/A100 runs
```

## Why Fused SiLU-Mul First

Fused SiLU-Mul is small enough to isolate, but real enough to matter:

- It appears on the LLaMA/Qwen SwiGLU MLP path.
- It directly tests manual Triton fusion against compiler fusion.
- It connects to the previous nano-vLLM profiling result where the steady
  decode path was dominated by attention and BF16 GEMM, leaving MLP activation
  fusion as a standalone follow-up.
- A valid conclusion does not require Triton to beat `torch.compile`; the goal
  is to identify when manual fusion is worth carrying.

## Quick Start

Install dependencies on a CUDA machine:

```bash
pip install -r requirements.txt
```

Run the phase-1 benchmark on the real GPU validation machine. Local Windows is
used for code editing, docs, git, and result integration; it is not used for
final CUDA/Triton performance conclusions.

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 50 --repeat 200 --no-write
```

Run profiler evidence on AutoDL RTX 3090 without writing trace files:

```bash
python studies/fused_silu_mul/profile.py --provider all --no-write
```

The benchmark reports latency, effective GB/s, speedup vs PyTorch unfused, gap
vs `torch.compile`, and max difference against:

```python
ref = torch.nn.functional.silu(gate.float()) * up.float()
```

When running on AutoDL or Colab, copy back the full terminal output, especially
the `BEGIN_BENCHMARK_CSV` / `END_BENCHMARK_CSV` block and profiler tables. The
summary documents are updated locally from that returned output instead of
committing generated result files from the cloud machine.
