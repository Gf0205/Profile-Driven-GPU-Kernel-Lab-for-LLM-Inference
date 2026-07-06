# 09 W4A16 GEMM

W4A16 量化矩阵乘法：权重以 INT4 存储，激活值保持 FP16，累加用 FP32。

## 原理

量化流程（离线预处理）：

```
scales  = (w_max - w_min) / 15
zeros   = -w_min / scales
w_q     = clamp(round(w / scales + zeros), 0, 15)
```

Kernel 内在线反量化：

```
b_fp32 = (b_int4 - zeros_exp) * scales_exp   # 每个 group 一组 scale/zero
b_fp16 = b_fp32.to(fp16)
acc    = tl.dot(a, b_fp16, acc=acc)
```

关键实现点：
- **权重打包**：8 个 INT4 压成 1 个 INT32（`w_packed |= w_q[i::8] << (i * 4)`）
- **Group 量化**：每 128 个 K 维度一组 scale/zero
- **在 Kernel 内解包**：避免反量化中间结果落 HBM
- `num_stages=4, num_warps=4` 提升 SM 利用率

## 在 Colab 中测什么

运行 `w4a16.py`，输出与 Python 反量化参考的对比：

| Shape (M, K, N, G) | 典型场景 |
|---------------------|---------|
| (1024, 512, 4096, 128) | 大型投影量化 |
| (1, 512, 4096, 128) | decode 单 token |
| (512, 4096, 4096, 128) | prefill 量化 GEMM |
| (64, 4096, 4096, 128) | 小 batch prefill |

**关注指标**：
1. `max_diff < 1.0`：与反量化参考一致
2. **量化噪声**：`(未量化输出 - W4A16输出).abs().max()` —— 这是 INT4 量化引入的误差，属于正常现象，通常 < 1.0

> **进阶压测**：加入 `triton.testing.do_bench` 对比 W4A16 vs FP16 GEMM 的 throughput 和显存占用（W4A16 权重显存节省约 4x）。
