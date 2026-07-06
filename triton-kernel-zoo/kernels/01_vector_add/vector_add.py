import torch
import triton
import triton.language as tl



# Colab第一个cell
#google colab 上运行前需要执行： !pip install triton -q
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Triton: {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 验证Triton能编译kernel
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device='cuda', dtype=torch.float16)
y = torch.randn(n, device='cuda', dtype=torch.float16)
out = torch.empty_like(x)
_add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)

# 验证正确性
assert torch.allclose(out, x + y, atol=1e-2)
print("✅ Triton kernel compiled and ran correctly!")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 24)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float16)
    y = torch.randn(size, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _add_kernel[(triton.cdiv(size, 1024),)](
                x, y, torch.empty_like(x), size, BLOCK=1024
            ), quantiles=quantiles
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: x + y, quantiles=quantiles
        )
    gbps = lambda ms: 3 * size * 2 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)