import triton
import torch
import triton.language as tl


@triton.jit
def layernorm_kernel(
    output_ptr:tl.tensor,input_ptr:tl.tensor,
    gamma_ptr:tl.tensor,beta_ptr:tl.tensor,stride_rows,
    n_cols,
    eps,
    BLOCK_SIZE:tl.constexpr,
):
    #计算当前是第几行
    row_idx = tl.program_id(0)

    #计算当前行的起始地址
    row_start_ptr = input_ptr + row_idx * stride_rows

    #计算行内偏移值
    row_offsets = tl.arange(0,BLOCK_SIZE)
    #建立刚性边界掩码，防止特征维度不是2的幂时发生非法物理跨界访问

    mask = row_offsets < n_cols
    #加载数据
    row = tl.load(row_start_ptr + row_offsets,mask=mask,other=0.0)


    #进行均值，方差，标准化计算
    mean = tl.sum(row,axis=0) / n_cols
    var = tl.sum((row - mean) * (row - mean),axis=0) / n_cols
    rsqrt_var = tl.math.rsqrt(var + eps)

    #加载gamma和beta
    gamma = tl.load(gamma_ptr + row_offsets,mask=mask,other=0.0)
    beta = tl.load(beta_ptr + row_offsets,mask=mask,other=0.0)

    #进行标准化计算
    layer_norm_output = (row - mean) * rsqrt_var * gamma + beta

    #存储相关结果

    output_start_ptr = output_ptr + row_idx * stride_rows
    tl.store(output_start_ptr + row_offsets,layer_norm_output,mask=mask)



def layernorm(x:torch.Tensor,gamma:torch.Tensor,beta:torch.Tensor,eps:float = 1e-5)->torch.Tensor:

    #获取输入向量的形状
    n_rows,n_cols = x.shape
    BLOCK_SIZE = 1 << ((n_cols - 1).bit_length())
    output = torch.empty_like(x)

    grid = (n_rows,)

    layernorm_kernel[grid](output,x,gamma,beta,x.stride(0),n_cols,eps,BLOCK_SIZE)

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
    print("Testing LayerNorm...")
    for rows, cols in cases:
        x = torch.randn(rows, cols, device='cuda', dtype=torch.float32)
        gamma = torch.randn(cols, device='cuda', dtype=torch.float32)
        beta  = torch.randn(cols, device='cuda', dtype=torch.float32)

        ref = torch.nn.functional.layer_norm(x, (cols,), gamma, beta)
        out = layernorm(x, gamma, beta)

        max_diff = (ref - out).abs().max().item()
        print(f"  ({rows:3d}, {cols:4d}): max_diff={max_diff:.2e} "
              f"{'✅' if max_diff < 5e-3 else '❌'}")


if __name__ == "__main__":
    test_correctness()