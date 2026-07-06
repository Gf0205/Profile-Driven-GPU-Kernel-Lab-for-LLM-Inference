# 06 Paged Decode Attention

Decode 阶段的 Attention，支持 **Paged KV Cache**（vLLM / SGLang 风格）。

## 原理

与 Prefill 不同，Decode 每次只处理一个 Query，但 KV Cache 被分块存放在不连续的物理块中：

1. 每个 program 处理一个 `(batch, head)` 对
2. 通过 `block_table` 查物理 block 编号
3. 遍历所有物理 KV block，累加 Online Softmax
4. BLOCK_Q = 1（Decode 特性）

KV Cache 布局：`[num_physical_blocks, num_heads, block_size, head_dim]`

## 在 Colab 中测什么

运行 `Paged Attention.py`，输出与逐 block 查表参考实现对比的误差：

| Batch Size | Heads | Head Dim | Block Size | Max Context |
|-----------|-------|----------|-----------|-------------|
| 4 | 8 | 64 | 16 | 128 |

**关注指标**：
- `max_diff < 0.05`（Decode 阶段误差容忍度稍高）
- 可调整 `max_ctx` 到 512/1024/2048 测试更长上下文

> **进阶压测**：可加入吞吐量测试，对比 naive decode attention 的加速比。
