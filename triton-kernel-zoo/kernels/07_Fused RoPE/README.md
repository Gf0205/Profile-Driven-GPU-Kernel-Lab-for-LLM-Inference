# 07 Fused RoPE

融合 Rotary Position Embedding，同时对 Q 和 K 做旋转，**一步到位，无需中间 tensor**。

## 原理

RoPE 旋转公式（复数乘法形式）：

```
x1 = x[..., :hd//2]
x2 = x[..., hd//2:]
out_x1 = x1 * cos - x2 * sin
out_x2 = x1 * sin + x2 * cos
```

融合优势：
- Q 和 K 在同一 kernel 内处理，避免中间 HBM 读写
- 预计算角度表 `angles = positions @ freqs`，只读一次
- Grid 三维分工：`(seq_len, head_groups, 2)`，2 专门区分 Q/K

## 在 Colab 中测什么

运行 `RoPE.py`，输出与手工 PyTorch 实现对比的 Q/K 旋转误差：

| Shape (seq, nq, nk, hd) | 典型场景 |
|-------------------------|---------|
| (256, 32, 8, 128) | GQA decode |
| (512, 32, 8, 128) | 中等序列 |
| (1024, 32, 8, 64) | 短 head_dim variant |

**关注指标**：
- `Q_diff < 0.01` 且 `K_diff < 0.01` → ✅
- 支持 GQA 场景（num_heads_q ≠ num_heads_k）
- 可扩展到 seq=4096/8192 测试长上下文 RoPE
