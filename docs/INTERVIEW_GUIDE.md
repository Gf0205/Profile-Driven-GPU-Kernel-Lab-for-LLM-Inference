# Interview Guide

This guide turns the repository into a concise, defensible technical narrative.
Every claim below is supported by the committed benchmark summaries. Raw cloud
logs and profiler traces are intentionally not committed.

## 90-Second Project Pitch

I built a profile-driven CUDA and Triton study around two LLM inference operator
classes: Tensor Core GEMM and fused SwiGLU activation. I used cuBLAS, eager
PyTorch, and `torch.compile` as strong baselines, validated correctness first,
then tested optimization hypotheses on preselected decode and prefill shapes.

For GEMM, profiling and code inspection suggested that the multi-warp WMMA
kernel repeatedly staged the same A and B data. I kept its 64x32 CTA mapping
fixed and introduced cooperative staging. That improved three representative
shapes by 1.32-1.58x, but the kernel remained 3.65-7.99x behind cuBLAS. The
result showed both the value of cross-warp reuse and the limitation of one fixed
kernel strategy against shape-aware library dispatch.

For SiLU-Mul, manual Triton reached 1.97x over eager PyTorch on a large prefill
shape, while eager remained best for tiny isolated calls. Profiler evidence
showed that `torch.compile` already emitted an equivalent fused Triton kernel
within about 2-6% of the manual kernel's GPU time. That led to a no-go on further
manual tuning: the practical boundary depends on shape and integration overhead,
not simply on whether an expression can be fused by hand.

## GEMM Deep Dive

### Problem

The initial scalar and tiled CUDA variants established a controlled progression
to WMMA Tensor Core execution. A 64x32 multi-warp CTA variant then behaved
differently across LLM shapes: it helped the wide Qwen up projection, was nearly
neutral on large prefill, and regressed on the long-K LLaMA down projection.

### Hypothesis

Each warp owned private A and B shared-memory tiles. Warps computing different N
tiles repeated A loads, and warps computing different M tiles repeated B loads.
The long K loop amplified this redundant staging.

The defensible wording is: repeated staging was an important performance
limitation in this kernel. It was not proven to be the only bottleneck.

### Controlled Change

The cooperative variant retained:

- the 64x32 CTA output tile
- eight warps per CTA
- the same warp-to-output mapping
- the same WMMA fragment computation

It changed shared staging to four unique A tiles and two unique B tiles per CTA,
with block-level synchronization before consumers loaded their fragments.

### Evidence

| Shape | Improvement over block-tiled WMMA |
|---|---:|
| `qwen_mlp_up_128` | 1.58x |
| `mlp_down_128` | 1.32x |
| `prefill_512_4096` | 1.42x |

Profiler kernel time independently reproduced the direction and magnitude of
the improvement. Correctness passed for every row.

### Why It Still Trails cuBLAS

The custom implementation uses one fixed tile, warp mapping, and simple K-loop.
Profiler kernel signatures indicated that cuBLAS selected different kernel
variants with different macro-tile characteristics by shape. A competitive
redesign would require architecture-specific kernel selection and a deeper
load/compute pipeline, approaching CUTLASS-level scope.

Do not claim that the profiler proved exact internal cuBLAS tile dimensions.
Kernel names only provide evidence of shape-specific variants and macro-tile
characteristics.

## Fused SiLU-Mul Deep Dive

### Question

When is a manual Triton fusion useful compared with eager PyTorch and a strong
`torch.compile` baseline?

### Result By Regime

- Tiny decode: eager wins isolated-call latency because all kernels take only
  about 1-2 us and runtime dispatch dominates.
- Large LLaMA prefill: Triton is 1.97x faster than eager by reducing two GPU
  kernels and an intermediate tensor to one fused kernel.
- Qwen prefill: Triton and eager are treated as tied because their p20-p80
  intervals overlap.

### What torch.compile Actually Did

Profiler output contained one `triton_poi_fused_mul_silu` kernel, so compiler
fusion worked. On the larger shapes, compiler-generated and manual Triton GPU
kernel times differed by only about 2-6%.

The benchmark compiled and warmed the function before timing. Therefore the
standalone-call gap is not initial compilation cost. It is consistent with
non-kernel invocation behavior such as guard/cache lookup, allocation, runtime
dispatch, submission cadence, and stream starvation, but the measurements do
not assign an exact microsecond cost to any one layer. Whole-graph compilation
may change the trade-off, so the project does not claim that manual Triton
generally beats `torch.compile`.

The final harness reports three timing views: isolated-call CUDA-event interval,
single-stream steady-state per-call interval under continuous asynchronous
submission, and profiler GPU kernel time. CUDA events do not directly measure
CPU wrapper time; host delays appear only when they leave the stream idle. Keep
the three views separate because none is a substitute for the others.

## Likely Questions

### Why did you stop optimizing GEMM?

The study had answered its hypothesis: cooperative staging consistently
improved the custom kernel. Closing the remaining cuBLAS gap would require a
different project scope involving architecture-specific tile dispatch,
multistage asynchronous pipelines, and deeper occupancy/register analysis. The
next change was no longer a narrow, evidence-backed ablation.

### Why is your GEMM slower than cuBLAS?

The goal was controlled bottleneck analysis, not replacing cuBLAS. The custom
kernel has fixed mapping and a simple pipeline; cuBLAS uses mature,
architecture-aware kernel variants selected for different M/N/K shapes.

### Did cooperative staging solve the bottleneck?

It removed an important limitation and improved all three preselected shapes.
It did not prove that staging was the only limitation. Synchronization, warp
scheduling, instruction pipeline depth, occupancy, and register behavior remain
possible contributors.

### Does Triton beat torch.compile?

Not as a general claim. The direct-call benchmark favored manual Triton, but the
generated GPU kernels were nearly equivalent. The difference was primarily in
the standalone runtime call path, and whole-graph compilation may change that
trade-off.

### Why can effective GB/s exceed RTX 4090 HBM bandwidth?

The metric counts logical minimum traffic while repeatedly using the same
tensors. Working sets can remain in the large L2 cache, so the number is an
effective throughput metric, not measured HBM bandwidth.

### What evidence is missing?

Nsight Compute hardware counters were unavailable on the rented AutoDL instance.
The project therefore does not claim measured occupancy, Tensor Core utilization,
or HBM transaction attribution. Its conclusions rely on correctness, controlled
ablation, CUDA-event statistics, source-level reasoning, and PyTorch profiler
kernel time.

## Defensible Resume Bullets

- Built a profile-driven CUDA/Triton performance study for realistic LLM
  inference shapes, with correctness gates, strong library/compiler baselines,
  repeat statistics, profiler attribution, and explicit go/no-go decisions.
- Developed FP16 WMMA Tensor Core GEMM variants with block tiling and
  cooperative shared-memory staging, improving three representative LLM shapes
  by 1.32-1.58x over the previous custom kernel.
- Compared custom GEMM against cuBLAS and used profiler evidence to attribute
  the remaining gap to fixed custom mapping versus shape-specific library kernel
  selection and a deeper Tensor Core pipeline.
- Implemented a fused Triton SiLU-Mul operator and measured up to 1.97x over
  eager PyTorch on large prefill, while showing that `torch.compile` generated
  a fused kernel within about 2-6% of manual Triton's GPU execution time.

## Claims To Avoid

- "Outperformed cuBLAS" or "cuBLAS-competitive GEMM"
- "Manual Triton is generally faster than torch.compile"
- "Proved exact internal cuBLAS tile sizes"
- "Measured occupancy or Tensor Core utilization"
- "Production-level CUTLASS, FlashAttention, W4A16, or TensorRT kernels"
