# 08 Fused SiLU-Mul (SwiGLU)

融合 SiLU (Sigmoid Linear Unit) 与 element-wise 乘法，是 LLaMA/Llama 架构 FFN 的核心算子。

## 原理

SwiGLU 激活函数：

```
gate_silu = gate / (1 + exp(-gate))
output = gate_silu * up
```

融合优势：
- 只做一次 HBM 读取（gate + up），一次写回
- 中间 `gate_silu` 不落显存
- 手写 SiLU 避免 `torch.nn.functional.silu` 的 kernel 切换开销

## 在 Colab 中测什么

运行 `silu.py`，分两步：

### 1. 正确性测试
| Shape | 场景 |
|-------|------|
| (1, 11008) | decode batch=1 |
| (32, 11008) | decode batch=32 |
| (1024, 11008) | prefill |
| (1, 8192) | 不同 FFN hidden |

**关注指标**：`max_diff < 0.01` → ✅

### 2. 性能压测（`benchmark.run`）
**关注指标**：
- 横轴：N（hidden_size），范围 {2048, 4096, 8192, 11008, 16384, 28672}
- 纵轴：GB/s
- 三条曲线对比：
  - **Triton Fused**（自研 kernel）
  - **PyTorch Unfused**（`silu(gate) * up`，两次 kernel）
  - **PyTorch Fused**（`torch.compile` 融合后）

理想情况：Triton Fused 明显优于 PyTorch Unfused，接近或超过 PyTorch Fused。
