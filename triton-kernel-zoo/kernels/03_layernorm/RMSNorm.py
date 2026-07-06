import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    output_ptr, input_ptr, gamma_ptr,
    stride_row,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * stride_row
    row_offsets = tl.arange(0,BLOCK_SIZE)
    mask = row_offsets < n_cols
    #加载数据
    row = tl.load(row_start_ptr + row_offsets,mask=mask,other=0.0)

    #计算L2均值
    L2_mean = tl.sum(row * row) / n_cols

    #计算rms值
    rms = tl.math.rsqrt(L2_mean + eps)

    #加载gamma
    gamma = tl.load(gamma_ptr + row_offsets,mask=mask,other=0.0)

    #计算标准化结果
    output = row * rms * gamma

    #存储结果
    output_start_ptr = output_ptr + row_idx * stride_row
    tl.store(output_start_ptr + row_offsets,output,mask=mask)

def rmsnorm(x:torch.Tensor,gamma:torch.Tensor,eps:float = 1e-5)->torch.Tensor:
    n_rows,n_cols = x.shape
    BLOCK_SIZE = 1 << ((n_cols - 1).bit_length())
    output = torch.empty_like(x)
    grid = (n_rows,)
    #启动kernel
    rmsnorm_kernel[grid](output,x,gamma,x.stride(0),n_cols,eps,BLOCK_SIZE)

    return output


def test_correctness():
    torch.manual_seed(42)
    cases = [
        (1,    512),
        (32,   512),
        (1,    896),    # Qwen2.5-0.5B hidden_size
        (24,   896),    # Qwen2.5-0.5B prefill
        (1,    4096),
        (64,   4096),
    ]
    print("Testing RMSNorm...")
    for rows, cols in cases:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        gamma = torch.randn(cols, device='cuda', dtype=torch.float32)

        # PyTorch reference: RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref = x * rms * gamma

        out = rmsnorm(x, gamma)

        max_diff = (ref - out).abs().max().item()
        print(f"  ({rows:3d}, {cols:4d}): max_diff={max_diff:.2e} "
              f"{'✅' if max_diff < 5e-3 else '❌'}")


if __name__ == "__main__":
    test_correctness()