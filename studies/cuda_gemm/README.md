# CUDA GEMM Optimization Study

CUDA GEMM is the first main line of this repository because it is the strongest
screening signal for GPU performance ownership. The goal is not just to
implement `C = A @ B`, but to run a controlled ablation:

```text
naive
-> shared-memory tiling
-> register blocking
-> vectorized load
-> Tensor Core / WMMA path
-> block-tiled multi-warp WMMA path
-> cooperative shared-tile WMMA path
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
| `cuda_wmma` | active | first Tensor Core path using WMMA fragments |
| `cuda_wmma_block_tiled` | active | multi-warp CTA tile, 64x32 C tile per block |
| `cuda_wmma_shared_tiles` | active | same 64x32 mapping with cooperative A/B staging across warps |

This study is now closed. The final optimization question was whether duplicate
per-warp global-to-shared staging explained the shape-dependent regression of
the block-tiled WMMA kernel. The cooperative-staging variant answered that
question without changing the CTA tile, warp count, or WMMA compute path.

## RTX 4090 Final Results

Official environment: NVIDIA GeForce RTX 4090, PyTorch `2.1.2+cu121`, FP16,
20 warmup iterations, and 100 measured repetitions. Latency is CUDA-event p50.
The three shapes were selected before implementing cooperative staging: one
positive case, one regression case, and one roughly neutral case.

| Shape (M x N x K) | cuBLAS ms | Block-tiled ms | Shared-tiles ms | Shared vs block | Shared TFLOP/s | Gap vs cuBLAS | rel_diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen up, 128 x 18944 x 4096 | 0.2004 | 1.1592 | **0.7322** | **1.58x** | 27.13 | 3.65x | 7.386e-4 |
| LLaMA down, 128 x 4096 x 11008 | 0.1160 | 1.2247 | **0.9267** | **1.32x** | 12.46 | 7.99x | 4.717e-4 |
| Prefill, 512 x 4096 x 4096 | 0.1116 | 0.7085 | **0.4997** | **1.42x** | 34.38 | 4.48x | 0.000e+0 |

All rows passed the established FP16 correctness check. Their p20/p50/p80
ranges were tight, so the improvement is not explained by timing noise.

## Profiler Attribution

PyTorch profiler independently reproduced the benchmark ordering using average
CUDA kernel time over 10 calls:

| Shape | Block-tiled kernel | Shared-tiles kernel | Profiler speedup |
|---|---:|---:|---:|
| `qwen_mlp_up_128` | 1204.1 us | 762.6 us | 1.58x |
| `mlp_down_128` | 1281.8 us | 969.6 us | 1.32x |
| `prefill_512_4096` | 803.9 us | 538.7 us | 1.49x |

The original 64x32 block kernel allocated private A and B staging tiles for
each of its eight warps. Warps sharing an M tile therefore loaded the same A
data twice, while warps sharing an N tile loaded the same B data four times on
every K-loop iteration. `cuda_wmma_shared_tiles` keeps the same output mapping
but loads four unique A tiles and two unique B tiles cooperatively per CTA.
The consistent 1.32-1.58x improvement validates duplicate staging as a real
bottleneck.

The cuBLAS-backed baseline dispatched different kernel signatures by shape:

| Shape | cuBLAS kernel macro-tile signature | Kernel time |
|---|---|---:|
| `qwen_mlp_up_128` | 256x128-like | 188.9 us |
| `mlp_down_128` | 128x64-like | 107.9 us |
| `prefill_512_4096` | 128x128-like | 109.4 us |

This is direct evidence that practical GEMM performance is shape-dependent.
The fixed 64x32 custom mapping improves when redundant staging is removed, but
it cannot reproduce cuBLAS's shape-aware kernel selection and deeper pipeline.

## Final Go/No-Go

**Go:** retain cooperative shared-memory staging as a successful architectural
optimization. It converts a profiler-backed hypothesis into a stable 1.3-1.6x
improvement on three preselected LLM inference shapes.

**No-go:** do not extend this study into an open-ended CUTLASS-style GEMM with
`cp.async`, multistage double buffering, warp specialization, or a broad tile
sweep. The best custom variant remains 3.65-7.99x slower than cuBLAS, and
`mlp_down_128` only recovers the block-tiled regression: its 0.9267 ms is
approximately tied with the original one-warp WMMA kernel at 0.9189 ms.

The result is not a claim of cuBLAS competitiveness. It is a controlled study
of how Tensor Cores, CTA mapping, memory hierarchy, and workload shape interact.

## AutoDL RTX 4090 Setup

The RTX 4090 screenshot configuration is suitable for this phase:

- GPU: RTX 4090 24GB, 1 card is enough.
- Driver/CUDA shown by AutoDL: driver `560.35.03`, CUDA `12.6`.
- Base image shown: `PyTorch / 2.1.0 / 3.10 / ubuntu22.04 / 12.1`.

Driver CUDA being newer than the image CUDA runtime is normal. Use the PyTorch
image first; only switch images if the CUDA extension build fails. For RTX 4090,
Ada architecture is `sm_89`; setting `TORCH_CUDA_ARCH_LIST` makes extension
build logs easier to interpret.

Start the instance, open the terminal, then run:

```bash
git clone https://github.com/Gf0205/Profile-Driven-GPU-Kernel-Lab-for-LLM-Inference.git
cd Profile-Driven-GPU-Kernel-Lab-for-LLM-Inference
python -m pip install --upgrade pip
pip install -r requirements.txt
export TORCH_CUDA_ARCH_LIST="8.9"
```

If `ninja` is missing for PyTorch CUDA extension builds:

```bash
pip install ninja
```

## Benchmark

Quick smoke test before the full run. This explicitly includes `cuda_naive` on
small shapes to validate correctness and the ablation baseline:

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 5 --repeat 10 --shapes decode_4096 decode_16_4096 --providers torch_matmul cuda_naive cuda_tiled cuda_reg_blocked cuda_vec4 cuda_wmma cuda_wmma_block_tiled --no-write
```

Default AutoDL RTX 4090 command. It prints all fields and writes no files.
It skips `cuda_naive` by default because naive GEMM on large prefill shapes can
waste cloud time without adding useful signal:

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --no-write
```

Tensor Core decision pass. This is the key next-stage question: does the WMMA
path significantly shrink the custom-vs-cuBLAS gap compared with scalar
tiling/register blocking?

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --providers torch_matmul cuda_reg_blocked cuda_vec4 cuda_wmma cuda_wmma_block_tiled --no-write
```

Block-tiled Tensor Core feasibility pass. This is the next stop/go checkpoint:
does moving from one-warp-per-16x16 tile to a multi-warp CTA tile improve large
prefill and MLP shapes?

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --shapes prefill_128_4096 prefill_512_4096 mlp_up_128 mlp_down_128 qwen_mlp_up_128 --providers torch_matmul cuda_wmma cuda_wmma_block_tiled --no-write
```

Hypothesis-driven diagnostic pass. This intentionally uses only three shapes:
`qwen_mlp_up_128` where 64x32 block tiling helps, `mlp_down_128` where it
regresses, and `prefill_512_4096` where it is roughly neutral/slightly positive.

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --shapes wmma_shape_diagnostic --providers torch_matmul cuda_wmma cuda_wmma_block_tiled cuda_wmma_shared_tiles --no-write
```

If the default command is stable, run a focused large-shape pass:

```bash
python studies/cuda_gemm/benchmark.py --dtype float16 --warmup 20 --repeat 100 --shapes prefill_128_4096 prefill_512_4096 mlp_up_128 mlp_down_128 --providers torch_matmul cuda_reg_blocked cuda_vec4 cuda_wmma cuda_wmma_block_tiled --no-write
```

Copy back:

- the `ENV ...` line
- every human-readable provider line
- the full `BEGIN_GEMM_CSV` / `END_GEMM_CSV` block
- any extension build error if compilation fails

Do not mix rows from different GPU models in one conclusion table. The official
results above are from RTX 4090 Ada (`sm_89`).

## Profiler

Profiler tables without writing trace files:

```bash
python studies/cuda_gemm/profiler.py --provider all --no-write
```

For a smaller profiler pass:

```bash
python studies/cuda_gemm/profiler.py --providers torch_matmul cuda_wmma cuda_wmma_block_tiled cuda_wmma_shared_tiles --shapes wmma_shape_diagnostic --no-write
```

PyTorch profiler shows launch/kernel timing but not enough architecture detail
for Tensor Core utilization or occupancy. To print Nsight Compute commands for
the three diagnostic shapes:

```bash
python studies/cuda_gemm/ncu_commands.py
```

Run the printed `ncu` commands only after the benchmark correctness pass is
clean. Copy back the terminal output; do not commit generated reports from
AutoDL.

Recommended attribution questions:

```text
qwen_mlp_up_128: why does 64x32 block tiling help wide-N up projection?
mlp_down_128: why does the same tile hurt down projection with large K?
prefill_512_4096: why is the gain only modest on square-ish large GEMM?
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
7. Cooperative-staging decision rule: compare `cuda_wmma_block_tiled` with
   `cuda_wmma_shared_tiles` while keeping the 64x32 CTA mapping fixed. Continue
   only if removing duplicate A/B staging gives a stable benefit on the three
   diagnostic shapes. Otherwise, conclude that simple cross-warp reuse is not
   the dominant missing mechanism and stop this GEMM phase before adding a
   deeper CUTLASS-style pipeline.
8. Avoid broad tile sweeps until the three-shape diagnostic explains the
   shape-dependent behavior. The next tile variants should be chosen from a
   concrete hypothesis, not parameter search.
