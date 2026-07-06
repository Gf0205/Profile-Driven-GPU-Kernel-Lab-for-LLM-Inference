import torch
import triton
import triton.language as tl

@triton.jit
def w4a16_gemm_kernel(
    A_ptr, B_ptr, Scales_ptr, Zeros_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_sm, stride_sn,
    stride_zm, stride_zn,
    stride_cm, stride_cn,
    group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       
    offs_k = tl.arange(0, BLOCK_K)                          
    packed_n_offs = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)  

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + packed_n_offs[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(NUM_K_BLOCKS):
        k_start = k_block * BLOCK_K
        k_abs = k_start + offs_k  

        mask_A = (offs_m[:, None] < M) & (k_abs[None, :] < K)
        a = tl.load(A_ptrs, mask=mask_A, other=0.0)  

        mask_B = (k_abs[:, None] < K) & (packed_n_offs[None, :] < (N // 8))
        b_packed = tl.load(B_ptrs, mask=mask_B, other=0)  

        # 👑 1. 权重解包：通过 reshape 展平成标准的二维大床
        shift = tl.arange(0, 8) * 4  
        b_packed_exp = b_packed[:, :, None]  
        b_int4 = (b_packed_exp >> shift[None, None, :]) & 0xF  
        b_int4 = tl.reshape(b_int4, (BLOCK_K, BLOCK_N))  

        group_id = k_start // group_size

        # 2. 加载并高精广播 scales
        scales_ptrs = Scales_ptr + group_id * stride_sm + offs_n * stride_sn
        scales = tl.load(scales_ptrs, mask=offs_n < N, other=0.0)  
        scales_exp = scales[None, :].to(tl.float32)  

        # 👑 3. 零点解包：和权重完全对称，直接 reshape 展平！消灭一切多余的转置！
        zeros_ptrs = Zeros_ptr + group_id * stride_zm + packed_n_offs * stride_zn
        z_packed = tl.load(zeros_ptrs, mask=packed_n_offs < (N // 8), other=0)  
        
        z_packed_exp = z_packed[:, None]  
        zeros_int4 = (z_packed_exp >> shift[None, :]) & 0xF  
        zeros_int4 = tl.reshape(zeros_int4, (BLOCK_N,))  
        zeros_exp = zeros_int4[None, :].to(tl.float32)  

        # 4. 反量化并累加矩阵乘
        b_fp32 = (b_int4.to(tl.float32) - zeros_exp) * scales_exp
        b_fp16 = b_fp32.to(tl.float16)
        
        acc = tl.dot(a, b_fp16, acc=acc)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, c, mask=C_mask)


def quantize_weight_w4(weight: torch.Tensor, group_size: int = 128):
    K, N = weight.shape
    assert N % 8 == 0, "N必须是8的倍数"
    assert K % group_size == 0, "K必须能被group_size整除"

    weight_f32 = weight.float()
    num_groups = K // group_size

    w_grouped = weight_f32.reshape(num_groups, group_size, N)

    w_min = w_grouped.min(dim=1).values  
    w_max = w_grouped.max(dim=1).values  

    scales = (w_max - w_min) / 15.0
    zeros_fp = -w_min / scales  

    w_q = torch.clamp(
        torch.round(w_grouped / scales.unsqueeze(1) + zeros_fp.unsqueeze(1)),
        0, 15
    ).to(torch.int32)  
    w_q_flat = w_q.reshape(K, N).contiguous()  

    # 隔空抽样打包权重
    w_packed = torch.zeros(K, N // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        w_packed |= w_q_flat[:, i::8] << (i * 4)

    zeros_int = torch.clamp(torch.round(zeros_fp), 0, 15).to(torch.int32).contiguous()
    
    # 隔空抽样打包零点
    zeros_packed = torch.zeros(num_groups, N // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        zeros_packed |= zeros_int[:, i::8] << (i * 4)

    return w_packed, scales.half(), zeros_packed


def w4a16_gemm(A: torch.Tensor, B_q: torch.Tensor, scales: torch.Tensor,
                zeros_q: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    M, K = A.shape
    K_w, N_packed = B_q.shape
    N = N_packed * 8

    C = torch.empty((M, N), dtype=torch.float16, device=A.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 32
    NUM_K_BLOCKS = triton.cdiv(K, BLOCK_K)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    w4a16_gemm_kernel[grid](
        A, B_q, scales, zeros_q, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_q.stride(0), B_q.stride(1),
        scales.stride(0), scales.stride(1),
        zeros_q.stride(0), zeros_q.stride(1),
        C.stride(0), C.stride(1),
        group_size,
        BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_BLOCKS,
        num_stages=4,   
        num_warps=4,    
    )
    return C


# ── 👑 你的完美 Python 参考反量化对账逻辑 ──
def dequantize_weight_w4(B_q, scales, zeros_q, K, N, group_size):    
    w_unpacked = torch.zeros(K, N, dtype=torch.int32, device=B_q.device)    
    for i in range(8):        
        w_unpacked[:, i::8] = (B_q >> (i * 4)) & 0xF    
    
    num_groups = K // group_size    
    zeros_unpacked = torch.zeros(num_groups, N, dtype=torch.int32, device=zeros_q.device)    
    for i in range(8):        
        zeros_unpacked[:, i::8] = (zeros_q >> (i * 4)) & 0xF    
    
    w_deq = torch.zeros(K, N, dtype=torch.float16, device=B_q.device)    
    for g in range(num_groups):        
        k_start = g * group_size        
        k_end = k_start + group_size        
        w_group = w_unpacked[k_start:k_end, :].float()   
        z = zeros_unpacked[g, :].float()                   
        s = scales[g, :].float()                           
        w_deq[k_start:k_end, :] = ((w_group - z) * s).half()    
    return w_deq

def test_correctness():    
    torch.manual_seed(42)    
    cases = [        
        (1024, 512,  4096, 128),        
        (1,    512,  4096, 128),        
        (512,  4096, 4096, 128),        
        (64,   4096, 4096, 128),    
    ]    
    print("🚀 验证 W4A16 kernel 与 Python 参考反量化的一致性...")    
    for M, K, N, G in cases:        
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)        
        W = torch.randn(K, N, device='cuda', dtype=torch.float16)        
        B_q, scales, zeros_q = quantize_weight_w4(W, group_size=G)        
        
        # Triton kernel 输出        
        out = w4a16_gemm(A, B_q, scales, zeros_q, group_size=G)        
        
        # Python 参考：相同的量化权重做反量化后矩阵乘        
        W_deq = dequantize_weight_w4(B_q, scales, zeros_q, K, N, G)        
        ref = (A.float() @ W_deq.float()).half()        
        
        max_diff = (ref - out).abs().max().item()        
        status = "✅" if max_diff < 1.0 else "❌"        
        print(f"  M={M:4d}, K={K:4d}, N={N:4d}, G={G:3d} | max_diff={max_diff:.4f} {status}")        
        
        # 额外显示量化本身引入的误差（供参考）        
        quant_ref = (A.float() @ W.float()).half()        
        quant_err = (quant_ref - out).abs().max().item()        
        print(f"    (vs 未量化原始权重的误差: {quant_err:.4f} ← 这是 INT4 量化噪声，不是 bug)")

if __name__ == "__main__":
    test_correctness()