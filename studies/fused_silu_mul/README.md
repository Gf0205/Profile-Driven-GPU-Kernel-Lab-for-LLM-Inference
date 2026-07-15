# Fused SiLU-Mul Operator Study

## Operator

Fused SiLU-Mul computes the activation path used inside SwiGLU MLPs:

```python
ref = torch.nn.functional.silu(gate.float()) * up.float()
```

The study compares:

1. `pytorch_unfused`: `torch.nn.functional.silu(gate) * up`
2. `torch_compile`: the same expression through `torch.compile`
3. `triton`: a single manual Triton kernel

The experiment asks one bounded question: when does a manual Triton fusion add
value over PyTorch eager execution, and when is `torch.compile` already the
stronger practical implementation? The Triton block size remains fixed at 1024;
this study is not a parameter sweep.

This study is now closed. The final result separates isolated operator-call
latency from GPU kernel execution time so that framework dispatch overhead is
not mistaken for kernel quality.

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

The official RTX 4090 preset contains six preselected shapes spanning one-token
decode, batched decode, medium/large prefill, and two intermediate dimensions.

## Benchmark

Run this correctness-first smoke test on AutoDL RTX 4090 after pulling the
latest code. It prints to the terminal and writes no files:

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 5 --repeat 20 --shapes llama7b_decode_b1 llama7b_prefill_128 --no-write
```

If every provider reports `correct=True`, run the official RTX 4090 sweep:

```bash
python studies/fused_silu_mul/benchmark.py --dtype float16 --warmup 25 --repeat 100 --amortized-inner 100 --shapes silu_official_rtx4090 --no-write
```

Optional BF16 run on GPUs with good BF16 support:

```bash
python studies/fused_silu_mul/benchmark.py --dtype bfloat16 --warmup 25 --repeat 100 --shapes silu_official_rtx4090 --no-write
```

Metrics:

- `latency_ms`: isolated-call p50 with one call per CUDA-event pair and a
  synchronization after every sample.
- `amortized_ms`: per-call p50 from 100 consecutive calls enclosed by one
  CUDA-event pair; `--amortized-inner` controls the batch size.
- isolated and amortized p20/p80/p95/p99 fields: repeat and tail statistics.
- `gbps`: effective lower-bound traffic, `(gate read + up read + output write)`.
- `amortized_gbps`: the same logical traffic metric using amortized latency.
- `speedup_vs_pytorch`: PyTorch unfused latency divided by provider latency.
- `gap_vs_torch_compile`: provider latency divided by `torch.compile` latency.
- `max_diff`, `rel_diff`, `correct`: correctness against the FP32 reference.

## Benchmark Contract

- Inputs are contiguous tensors from `torch.randn` with seed 0. Official rows
  use FP16 and preselected LLaMA/Qwen intermediate dimensions.
- Eager, compiled, and Triton paths receive the same tensors. Output allocation
  is part of every provider call.
- The `torch.compile` function is created and executed once before either timing
  region; initial graph compilation is excluded.
- CUDA events measure elapsed time on one CUDA stream; they do not directly
  measure Python or CPU wrapper time. Host-side submission/allocation delays are
  reflected only when they leave the stream idle between the two events.
- Isolated timing uses one call per event pair. Amortized timing reports the
  single-stream steady-state per-call interval from 100 continuous asynchronous
  submissions. PyTorch profiler provides the third view: GPU kernel execution
  time.
- Correctness uses `silu(gate.float()) * up.float()` with FP16 tolerances
  `atol=0.02`, `rtol=0.002`.
- Inputs are reused across repeats, so effective GB/s may reflect L2 residency
  and is not measured HBM bandwidth.

Copy back:

- the human-readable per-shape lines
- the full `BEGIN_BENCHMARK_CSV` / `END_BENCHMARK_CSV` block
- device name, PyTorch version, and Triton version if printed by the environment

## Profiler

Print profiler evidence for one decode and one prefill shape without writing
trace files:

```bash
python studies/fused_silu_mul/profiler.py --providers pytorch_unfused torch_compile triton --shapes silu_profile_diagnostic --no-write
```

If a Chrome trace is explicitly needed on the cloud machine, omit `--no-write`
and provide an output directory:

```bash
python studies/fused_silu_mul/profiler.py --providers pytorch_unfused torch_compile triton --shapes silu_profile_diagnostic --output-dir studies/fused_silu_mul/results/profiler_rtx4090
```

Expected analysis questions:

- Does PyTorch unfused launch separate activation/multiply kernels?
- Does `torch.compile` fuse the elementwise expression into one generated kernel?
- Is the manual Triton kernel launch overhead visible for tiny decode shapes?
- For larger prefill shapes, is throughput close to a memory bandwidth limit?

## Analysis Rules

1. Correctness first: if `max_diff` is outside a reasonable FP16/BF16 tolerance,
   do not discuss performance for that row.
2. Keep the three baselines separate: PyTorch unfused, `torch.compile`, and
   Triton fused. A Triton win over only PyTorch unfused is not enough to claim
   the best practical implementation.
3. Analyze by shape: decode-like small token counts and prefill-like larger
   token counts should be discussed separately.
4. Use all performance fields: latency, GB/s, speedup vs PyTorch unfused, gap
   vs `torch.compile`, and p20/p50/p80 repeat statistics.
5. Interpret small-shape variance as possible launch overhead or timing noise.
   Interpret large-shape saturation as a memory-bandwidth question before
   blindly tuning `BLOCK_SIZE`.
6. Go/no-go: continue optimizing Triton only if it is stable across multiple
   realistic shapes against `torch.compile`. If it mainly beats PyTorch unfused
   but loses to `torch.compile`, the conclusion is still valid: manual Triton
   fusion is educational and validates fusion mechanics, while compiler fusion
   is the stronger practical baseline for this operator.

## Results Summary

Official environment: NVIDIA GeForce RTX 4090, PyTorch `2.1.2+cu121`, Triton
`2.1.0`, CUDA `12.1`, FP16, 25 warmup iterations, and 100 measured repetitions.
All providers passed correctness against the FP32 reference.

| Shape | Regime | PyTorch ms | compile ms | Triton ms | Triton vs PyTorch | Triton / compile | Triton GB/s | Triton rel_diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `llama7b_decode_b1` | decode | **0.0213** | 0.0625 | 0.0366 | 0.58x | 0.59x | 1.80 | 3.290e-4 |
| `llama7b_decode_b16` | decode | **0.0206** | 0.0635 | 0.0358 | 0.57x | 0.56x | 29.49 | 2.631e-4 |
| `llama7b_prefill_128` | prefill | **0.0205** | 0.0625 | 0.0376 | 0.54x | 0.60x | 224.56 | 3.360e-4 |
| `llama7b_prefill_1024` | prefill | 0.0809 | 0.0625 | **0.0410** | **1.97x** | **0.66x** | 1651.20 | 2.491e-4 |
| `qwen_like_decode_b1` | decode | **0.0206** | 0.0616 | 0.0364 | 0.57x | 0.59x | 3.13 | 3.547e-4 |
| `qwen_like_prefill_512` | prefill | 0.0389 | 0.0622 | 0.0386 | 1.01x | 0.62x | 1506.73 | 2.448e-4 |

`Triton / compile` is a latency ratio, so values below 1.0 favor the manual
Triton call path. In the isolated protocol, each CUDA-event pair encloses one
call and is followed by synchronization. Host delays affect the event interval
only when they starve the stream of work. The result must not be presented as
either direct CPU wrapper time or pure kernel execution time.

The GB/s field uses logical minimum traffic: two input reads and one output
write. Values above RTX 4090 peak HBM bandwidth are possible because the same
tensors are reused across repetitions and the working set can be L2-resident.
These are effective throughput values, not proof of HBM saturation.

## Profiler Evidence

PyTorch profiler shows the actual GPU work per operator call:

| Shape | PyTorch unfused | torch.compile fused | Triton fused |
|---|---:|---:|---:|
| `llama7b_decode_b1` | 2 kernels, 1.95 us total | 1 kernel, 1.00 us | 1 kernel, 1.00 us |
| `llama7b_prefill_1024` | 2 kernels, 74.35 us total | 1 kernel, 15.00 us | 1 kernel, 14.20 us |
| `qwen_like_prefill_512` | 2 kernels, 38.65 us total | 1 kernel, 13.00 us | 1 kernel, 12.65 us |

The eager path launches separate SiLU and multiply kernels. `torch.compile`
emits one `triton_poi_fused_mul_silu` kernel, proving that compiler fusion is
working. On the two larger profiler shapes, its generated kernel is within
about 2-6% of the manual Triton kernel. The larger standalone-call difference is
consistent with non-kernel invocation behavior around the compiled callable,
but CUDA-event and profiler runs cannot assign an exact number of microseconds
to wrapper, guard/cache lookup, allocation, dispatch, or queue starvation.

For decode and smaller prefill shapes, all GPU kernels take only about 1-2 us.
The eager path wins isolated-call latency because framework/dispatch overhead
dominates the single fused kernel's saved memory traffic. At 1024 LLaMA tokens,
the data path is large enough for fusion to matter: Triton is 1.97x faster than
eager, while `torch.compile` is 1.29x faster than eager. The 1.01x Qwen prefill
result is within the overlapping p20-p80 intervals and is treated as a tie, not
as a Triton win.

## Go/No-Go Criteria

Manual Triton fusion is a go when it gives a stable advantage for relevant
decode/prefill shapes or exposes behavior that `torch.compile` cannot reliably
cover. It is a no-go when `torch.compile` already emits an equivalent fused
kernel with equal or better latency and less maintenance cost.

## Final Go/No-Go

**Go:** retain the Triton implementation as a clear demonstration of fusion
mechanics and as a useful direct-call path for sufficiently large prefill
shapes. It removes an intermediate tensor and reduces two GPU kernels to one.

**No-go:** do not continue tuning this kernel or claim that manual Triton is
generally superior to `torch.compile`. The compiler already emits an equivalent
fused Triton kernel with comparable observed GPU execution time. Its standalone
compiled callable has a higher measured direct-call interval in this PyTorch
2.1.2 microbenchmark, but the cause cannot be reduced to wrapper time alone and
whole-graph compilation may change the trade-off.

The practical boundary is shape- and integration-dependent: eager execution is
best for tiny isolated calls, while fusion becomes valuable once memory traffic
dominates. Manual Triton ownership is justified when direct-call control or a
larger unsupported fusion is required, not merely because the expression can be
written by hand.
