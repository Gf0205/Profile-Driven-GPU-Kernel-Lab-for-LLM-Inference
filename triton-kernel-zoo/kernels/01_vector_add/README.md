# 01 Vector Add

入门级 Triton Kernel：两个向量逐元素相加。

## 原理

每个 CUDA thread 负责一个元素的加法：

```python
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

- `tl.program_id(0)` 获取全局 thread block ID（即第几个 block）
- 每个 block 处理 BLOCK 个元素
- 通过 mask 处理尾部不整除的边界

## 在 Colab 中测什么

运行 `vector_add.py`，会输出 PyTorch 和 Triton 在不同规模下的 **GB/s 吞吐曲线**（`benchmark.run(print_data=True, show_plots=True)`）。

测试规模：`size ∈ {2^12, 2^13, ..., 2^23}`（约 4K ~ 8M 元素）。

**关注指标**：
- 横轴：向量长度（对数）
- 纵轴：GB/s（越大越好）
- 对比：Triton vs PyTorch 原生 `x + y`
