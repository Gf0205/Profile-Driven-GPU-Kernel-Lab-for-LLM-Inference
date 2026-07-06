# 03 LayerNorm / RMSNorm

对最后维度做 Layer Normalization 和 RMS Normalization。

## 原理

**LayerNorm**（含 gamma/beta 可学习参数）：

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

**RMSNorm**（去掉 mean 偏移，更高效）：

```
y = x * rsqrt(mean(x^2) + eps) * gamma
```

每个 `program_id(0)` 处理一行，BLOCK_SIZE 对齐到 `2^bit_length(n_cols)`。

## 在 Colab 中测什么

运行 `layernorm.py` 和 `RMSNorm.py`，输出与 PyTorch 参考实现对比的误差：

| Rows | Cols | 典型场景 |
|------|------|---------|
| 1 | 512 | decode 单 token |
| 32 | 512 | decode batch |
| 1 | 896 | Qwen2.5-0.5B hidden_size |
| 24 | 896 | prefill |
| 1 | 4096 | LLaMA hidden_size |
| 64 | 4096 | prefill batch |

**关注指标**：
- `max_diff < 5e-3` → ✅
- 重点关注 896 和 4096（真实 LLM hidden size）
