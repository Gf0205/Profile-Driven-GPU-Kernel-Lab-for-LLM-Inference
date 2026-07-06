import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride,   # 每行之间的stride（对于连续内存 = n_cols）
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # ---- 填空1 ----
    # 获取当前program的id，它代表第几行
    row_idx = tl.program_id(0)
    
    # ---- 填空2 ----
    # 计算这一行的输入起始指针
    # 提示：input_ptr是整个矩阵的起始地址，row_idx行的起始 = ?
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # ---- 填空3 ----
    # 生成列偏移量，从0到BLOCK_SIZE-1
    col_offsets = tl.arange(0,BLOCK_SIZE)
    
    # ---- 填空4 ----
    # 生成mask：只有col_offsets < n_cols的位置才是有效数据
    mask = col_offsets < n_cols
    
    # ---- 填空5 ----
    # 从HBM加载这一行数据
    # 无效位置用 -float('inf') 填充（为什么是-inf？想想对softmax的影响）
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # ---- 填空6 ----
    # 数值稳定的softmax三步：
    # step1: 求行最大值
    row_max = tl.max(row,axis=0)
    # step2: 减最大值后求exp
    numerator = tl.exp(row - row_max)
    # step3: 求和
    denominator = tl.sum(numerator,axis=0)
    # step4: 归一化
    softmax_output = numerator / denominator
    
    # ---- 填空7 ----
    # 计算输出指针并写回
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptr_offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output_row_start_ptr + output_ptr_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    
    # ---- 填空8 ----
    # BLOCK_SIZE需要是2的幂且能容纳整行
    # triton.next_power_of_2(n_cols) 可以帮你计算
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty_like(x)
    
    # ---- 填空9 ----
    # grid应该是什么？每个program处理一行，共M行
    softmax_kernel[(n_rows,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


# 正确性验证——这部分我帮你写好了，你只需要跑通上面的部分
def test_correctness():
    torch.manual_seed(42)
    test_cases = [
        (128, 64),
        (512, 128),
        (1024, 256),
        (256, 1024),
    ]
    print("Testing softmax correctness...")
    for shape in test_cases:
        x = torch.randn(shape, device='cuda', dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        out = softmax(x)
        max_diff = (ref - out).abs().max().item()
        status = "✅" if max_diff < 1e-5 else "❌"
        print(f"  shape={str(shape):>12s}: max_diff={max_diff:.2e} {status}")

if __name__ == "__main__":
    test_correctness()