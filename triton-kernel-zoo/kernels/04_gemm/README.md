# 04 GEMM

通用矩阵乘法 `C = A @ B`，是 LLM 中全连接层的核心。

## 原理

采用 Blocked GEMM 策略：

1. 将 C 划分为 `BLOCK_M × BLOCK_N` 的 tile
2. 每个 program 处理一个 tile
3. K 方向循环累加，用 `tl.dot` 触发 Tensor Core

关键配置：
- `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`
- 输出累加器用 `tl.float32`，最后转 `float16`

## 在 Colab 中测什么

运行 `GEMM.py`，输出正确性测试：

| Shape (M, N, K) | 典型场景 |
|----------------|---------|
| (64, 64, 64) | 小型投影 |
| (128, 256, 64) | MLP intermediate |
| (512, 1024, 512) | 大型投影 |
| (1, 4096, 4096) | decode 单 token FFN |
| (128, 4096, 4096) | prefill batch |

**关注指标**：
- `max_diff`：绝对误差
- `rel`：相对误差（相对于 ref 最大值），< 1% 为 ✅
