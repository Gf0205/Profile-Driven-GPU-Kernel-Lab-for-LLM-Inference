# 02 Softmax

对最后维度做数值稳定的 Softmax，**含填空练习**，适合作为第一个进阶 kernel。

## 原理

对每一行做：

1. 减行最大值（数值稳定）
2. 取 exp
3. 求和
4. 归一化

每个 `program_id(0)` 处理一行，BLOCK_SIZE 对齐到 `next_power_of_2(n_cols)`。

## 在 Colab 中测什么

运行 `softmax.py`，输出正确性测试结果：

| Shape | 预期最大误差 |
|-------|------------|
| (128, 64) | < 1e-5 |
| (512, 128) | < 1e-5 |
| (1024, 256) | < 1e-5 |
| (256, 1024) | < 1e-5 |

**关注指标**：每种 shape 的 `max_diff`，确认与 PyTorch `torch.softmax` 一致。
