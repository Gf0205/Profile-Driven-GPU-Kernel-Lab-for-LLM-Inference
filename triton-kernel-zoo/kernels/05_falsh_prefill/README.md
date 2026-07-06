# 05 Flash Attention Prefill

Prefill 阶段的全量 attention kernel，实现了 **Online Softmax** 避免溢出。

## 原理

核心优化：
- **Online Softmax**：`m_i, l_i` 状态标量替代完整 `exp(S)` 矩阵
- **Block 循环**：K/V 分 BLOCK_KV 块加载，逐步累加
- **Causal Mask**：用全局绝对坐标在 scores 上加三角 mask
- **V 指针转置**：`[HEAD_DIM, BLOCK_KV]` 布局完美对齐 `tl.dot`

Grid：`num_heads × num_query_blocks`，每个 program 处理一个 (head, query_block)。

## 在 Colab 中测什么

运行 `FlashAttention Prefill.py`，输出与 PyTorch 标准 Attention 的对比：

| Shape (seq, heads, head_dim) | 典型场景 |
|------------------------------|---------|
| (256, 8, 64) | 短序列 prefill |
| (512, 8, 64) | 中等序列 prefill |

**关注指标**：
- `max_diff < 0.01`（万分之一，允许半精度累加误差）
- 可扩展到 (2048, 32, 64)、(4096, 32, 128) 测试更长序列

> **进阶压测**：可加上 `triton.testing.do_bench` 测量 throughput (TFLOPS)，与 `flash-attn` 库对比。
