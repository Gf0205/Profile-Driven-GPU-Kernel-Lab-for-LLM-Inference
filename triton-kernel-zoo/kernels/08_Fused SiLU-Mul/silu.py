import triton
import torch
import triton.language as tl

@triton.jit
def swiglu_hardcore_kernel(
    output_ptr, gate_ptr, up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # 1. 搬运数据
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0)

    # 2. 强转 fp32 保障数值稳定性
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)

    # 3. 手工实现 SiLU-Mul
    exp_neg_gate = tl.exp(-gate)
    gate_silu = gate / (1.0 + exp_neg_gate)  # 用1.0更规范，避免隐式类型转换风险
    result = gate_silu * up

    # 4. 写回
    tl.store(output_ptr + offs, result.to(tl.float16), mask=mask)


def swiglu_hardcore(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    n = gate.numel()

    BLOCK_SIZE = 1024
    # ✅ 修复点1：用triton模块的宿主端cdiv，不用tl.cdiv
    grid = triton.cdiv(n, BLOCK_SIZE)

    # ✅ 修复点2：网格参数用元组形式，符合Triton多维网格规范
    swiglu_hardcore_kernel[(grid,)](out, gate, up, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def test_correctness():
    torch.manual_seed(42)
    shapes = [
        (1,    11008),   # batch=1,  decode phase
        (32,   11008),   # batch=32, decode phase
        (1024, 11008),   # prefill
        (1,    8192),    
    ]
    print("🚀 开始引爆纯手工硬核 SwiGLU Triton 算子...")
    for shape in shapes:
        gate = torch.randn(shape, device='cuda', dtype=torch.float16)
        up   = torch.randn(shape, device='cuda', dtype=torch.float16)

        ref = torch.nn.functional.silu(gate.float()) * up.float()
        ref = ref.half()

        out = swiglu_hardcore(gate, up)
        diff = (ref - out).abs().max().item()
        print(f"  shape={str(shape):>14s}: max_diff={diff:.2e} {'✅' if diff < 0.01 else '❌'}")


# ── 核心 Benchmark ──
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2048, 4096, 8192, 11008, 16384, 28672],
        line_arg='provider',
        line_vals=['triton_fused', 'torch_unfused', 'torch_fused'],
        line_names=['Triton Fused', 'PyTorch Unfused', 'PyTorch Fused'],
        styles=[('blue', '-'), ('red', '--'), ('green', '-.')],
        ylabel='GB/s',
        plot_name='silu-mul-fusion-hardcore',
        args={'M': 1024},
    )
)
def benchmark(M, N, provider):
    gate = torch.randn(M, N, device='cuda', dtype=torch.float16)
    up   = torch.randn(M, N, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton_fused':
        ms, mn, mx = triton.testing.do_bench(lambda: swiglu_hardcore(gate, up), quantiles=quantiles)
    elif provider == 'torch_unfused':
        ms, mn, mx = triton.testing.do_bench(
            lambda: torch.mul(torch.nn.functional.silu(gate), up), 
            quantiles=quantiles
        )
    else:  # torch_fused
        compiled_fn = torch.compile(lambda g, u: torch.nn.functional.silu(g) * u)
        _ = compiled_fn(gate, up) 
        ms, mn, mx = triton.testing.do_bench(lambda: compiled_fn(gate, up), quantiles=quantiles)

    gbps = lambda ms: (2 + 1) * M * N * 2 / ms * 1e-6  
    return gbps(ms), gbps(mx), gbps(mn)

if __name__ == "__main__":
    test_correctness()
    print("\n⚡️ 启动工业级吞吐量 (GB/s) 压榨决战...")
    benchmark.run(print_data=True, show_plots=True)