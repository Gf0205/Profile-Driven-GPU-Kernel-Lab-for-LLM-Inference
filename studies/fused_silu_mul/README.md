# Phase 1: Fused SiLU-Mul Operator Study

## Operator

Fused SiLU-Mul computes the activation path used inside SwiGLU MLPs:

```python
ref = torch.nn.functional.silu(gate.float()) * up.float()
```

The study compares:

1. `pytorch_unfused`: `torch.nn.functional.silu(gate) * up`
2. `torch_compile`: the same expression through `torch.compile`
3. `triton`: a single manual Triton kernel

## Shape Sweep

The benchmark uses decode and prefill regimes over LLaMA/Qwen-like intermediate
sizes:

| Shape name | Tokens | Intermediate | Regime |
|---|---:|---:|---|
| llama7b_decode_b1 | 1 | 11008 | decode |
| llama7b_decode_b16 | 16 | 11008 | decode |
| llama7b_decode_b32 | 32 | 11008 | decode |
| llama7b_prefill_128 | 128 | 11008 | prefill |
| llama7b_prefill_1024 | 1024 | 11008 | prefill |
| llama13b_decode_b1 | 1 | 13824 | decode |
| llama13b_prefill_512 | 512 | 13824 | prefill |
| qwen_like_decode_b1 | 1 | 18944 | decode |
| qwen_like_decode_b16 | 16 | 18944 | decode |
| qwen_like_prefill_512 | 512 | 18944 | prefill |
| wide_mlp_decode_b1 | 1 | 28672 | decode |
| wide_mlp_prefill_256 | 256 | 28672 | prefill |

## Benchmark

T4:

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 25 --repeat 100 --output studies/fused_silu_mul/results/t4_fused_silu_mul.csv
```

RTX 3090:

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 50 --repeat 200 --output studies/fused_silu_mul/results/rtx3090_fused_silu_mul.csv
```

Optional BF16 run on GPUs with good BF16 support:

```bash
python studies/fused_silu_mul/benchmark.py --dtype bfloat16 --output studies/fused_silu_mul/results/rtx3090_bf16_fused_silu_mul.csv
```

Metrics:

- `latency_ms`: median CUDA-event latency after warmup.
- `gbps`: effective lower-bound traffic, `(gate read + up read + output write)`.
- `speedup_vs_pytorch`: PyTorch unfused latency divided by provider latency.
- `gap_vs_torch_compile`: provider latency divided by `torch.compile` latency.
- `max_diff`: absolute max difference against the FP32 correctness reference.

## Profiler

Capture profiler evidence for one decode and one prefill shape:

```bash
python studies/fused_silu_mul/profile.py --provider all --output-dir studies/fused_silu_mul/results/profiler_t4
```

Expected analysis questions:

- Does PyTorch unfused launch separate activation/multiply kernels?
- Does `torch.compile` fuse the elementwise expression into one generated kernel?
- Is the manual Triton kernel launch overhead visible for tiny decode shapes?
- For larger prefill shapes, is throughput close to a memory bandwidth limit?

## Results Summary

After a benchmark run, generate a Markdown table:

```bash
python studies/fused_silu_mul/summarize_results.py studies/fused_silu_mul/results/t4_fused_silu_mul.csv --output studies/fused_silu_mul/results/t4_summary.md
```

Paste the generated table here when promoting a run into the project README.
Keep the raw CSV in `results/` as the source of truth.

| Shape | Regime | PyTorch ms | compile ms | Triton ms | Triton speedup vs PyTorch | Triton gap vs compile | Triton max_diff |
|---|---|---:|---:|---:|---:|---:|---:|
| pending T4/3090 run | - | - | - | - | - | - | - |

## Go/No-Go Criteria

Manual Triton fusion is a go when it gives a stable advantage for relevant
decode/prefill shapes or exposes behavior that `torch.compile` cannot reliably
cover. It is a no-go when `torch.compile` already emits an equivalent fused
kernel with equal or better latency and less maintenance cost.
