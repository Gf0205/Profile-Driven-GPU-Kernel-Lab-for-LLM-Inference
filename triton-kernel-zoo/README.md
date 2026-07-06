# Triton Kernel Zoo

使用 Triton 实现的高性能 GPU kernel 集合，涵盖 LLM 推理中从向量运算到量化 GEMM 的核心算子。每个 kernel 附带正确性验证和性能压测，支持在 Google Colab T4 / RTX 3090 上运行。

## 目录结构

```
triton-kernel-zoo/
├── kernels/
│   ├── 01_vector_add/          # 向量加法 (入门级)
│   ├── 02_softmax/             # Softmax (含填空练习)
│   ├── 03_layernorm/           # LayerNorm + RMSNorm
│   ├── 04_gemm/                # 通用矩阵乘法
│   ├── 05_flash_prefill/       # Flash Attention Prefill
│   ├── 06_paged_decode_attn/   # Paged KV Cache Decode Attention
│   ├── 07_fused_rope/          # Fused Rotary Position Embedding
│   ├── 08_fused_silu_mul/      # Fused SiLU-Mul (SwiGLU FFN 核心)
│   └── 09_w4a16_gemm/          # W4A16 量化 GEMM
└── README.md
```

## Kernel 总览

| # | Kernel | 难度 | 在 Colab 中测什么指标 |
|---|--------|------|----------------------|
| 01 | `vector_add` | 入门 | GB/s 吞吐（规模 2^12 ~ 2^24） |
| 02 | `softmax` | 初级 | 最大误差 vs PyTorch（4 种 shape） |
| 03 | `layernorm` + `RMSNorm` | 初级 | 最大误差 vs PyTorch（6 种 shape） |
| 04 | `GEMM` | 中级 | 最大/相对误差 vs `torch.matmul` |
| 05 | `FlashAttention Prefill` | 高级 | 最大误差 vs 标准 Attention |
| 06 | `Paged Decode Attention` | 高级 | 最大误差 vs 逐 block 查表参考实现 |
| 07 | `Fused RoPE` | 高级 | Q/K 旋转后最大误差 vs 手工实现 |
| 08 | `Fused SiLU-Mul` | 中级 | GB/s 吞吐（Triton vs PyTorch fused/unfused） |
| 09 | `W4A16 GEMM` | 高级 | 与反量化参考实现的误差，量化噪声分析 |

## 环境

```bash
# Google Colab (推荐 T4 GPU)
!pip install triton -q

# 本地
# - GPU: NVIDIA RTX 3090 (24GB) / T4 / A100
# - CUDA: 12.x
# - Triton: latest
```

## 如何使用

每个 kernel 目录下都有独立的 `README.md` 说明该算子的原理和 Colab 指标测试方法。直接运行对应的 Python 文件即可：

```bash
python kernels/04_gemm/GEMM.py        # 正确性测试
python kernels/08_fused_silu_mul/silu.py  # 性能压测
```
