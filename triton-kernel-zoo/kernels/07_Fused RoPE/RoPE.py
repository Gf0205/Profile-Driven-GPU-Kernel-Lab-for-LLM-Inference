import torch
import triton
import triton.language as tl

@triton.jit
def fused_rope_kernel(
    Q_ptr, K_ptr,
    OutQ_ptr, OutK_ptr,
    Freqs_ptr,              # 预计算的角度表，shape=[max_seq_len, head_dim//2]
    stride_qs, stride_qh, stride_qd,      
    stride_ks, stride_kh, stride_kd,      
    stride_oqs, stride_oqh, stride_oqd,
    stride_oks, stride_okh, stride_okd,
    stride_fs, stride_fd,            
    num_heads_q,
    num_heads_k,
    HEAD_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,       # 每个program同时处理多少个head
):
    seq_id = tl.program_id(0)
    head_group = tl.program_id(1)
    qk_flag = tl.program_id(2)

    # 简化分支选择
    if qk_flag == 0:
        base_ptr   = Q_ptr
        out_ptr    = OutQ_ptr
        stride_s   = stride_qs
        stride_h   = stride_qh
        stride_d   = stride_qd
        stride_os  = stride_oqs
        stride_oh  = stride_oqh
        stride_od  = stride_oqd
        num_heads  = num_heads_q
    else:
        base_ptr   = K_ptr
        out_ptr    = OutK_ptr
        stride_s   = stride_ks
        stride_h   = stride_kh
        stride_d   = stride_kd
        stride_os  = stride_oks
        stride_oh  = stride_okh
        stride_od  = stride_okd
        num_heads  = num_heads_k

    head_start = head_group * BLOCK_HEADS
    heads = head_start + tl.arange(0, BLOCK_HEADS)
    head_mask = heads < num_heads

    # ✅ 1. 铁血修复点：直接在 arange 内部融入 constexpr 运算，砸碎 ValueError！
    freq_offs = tl.arange(0, HEAD_DIM // 2)
    freq_ptrs = Freqs_ptr + seq_id * stride_fs + freq_offs * stride_fd
    angles = tl.load(freq_ptrs)

    cos_val = tl.cos(angles)
    sin_val = tl.sin(angles)

    # 精准分割前后半段特征空间（[0 ~ hd//2] 和 [hd//2 ~ hd]）
    offs_d1 = tl.arange(0, HEAD_DIM // 2)
    offs_d2 = tl.arange(HEAD_DIM // 2, HEAD_DIM)

    # 构造并线加载指针大盘 [BLOCK_HEADS, half_dim]
    ptrs_x1 = base_ptr + seq_id * stride_s + heads[:, None] * stride_h + offs_d1[None, :] * stride_d
    ptrs_x2 = base_ptr + seq_id * stride_s + heads[:, None] * stride_h + offs_d2[None, :] * stride_d

    x1 = tl.load(ptrs_x1, mask=head_mask[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(ptrs_x2, mask=head_mask[:, None], other=0.0).to(tl.float32)

    # 旋转矩阵复数变换一气呵成：[BLOCK_HEADS, half_dim]
    out_x1 = x1 * cos_val[None, :] - x2 * sin_val[None, :]
    out_x2 = x1 * sin_val[None, :] + x2 * cos_val[None, :]

    out_ptrs_x1 = out_ptr + seq_id * stride_os + heads[:, None] * stride_oh + offs_d1[None, :] * stride_od
    out_ptrs_x2 = out_ptr + seq_id * stride_os + heads[:, None] * stride_oh + offs_d2[None, :] * stride_od

    # ✅ 2. 铁血修复点：写回时的 mask 必须广播至二维 [BLOCK_HEADS, None]，与输入完美对称！
    tl.store(out_ptrs_x1, out_x1.to(tl.float16), mask=head_mask[:, None])
    tl.store(out_ptrs_x2, out_x2.to(tl.float16), mask=head_mask[:, None])

def precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0, device='cuda'):
    half = head_dim // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2 * i / head_dim))          
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)               
    return angles  

def rope(q, k, angles):
    seq_len, num_heads_q, head_dim = q.shape
    num_heads_k = k.shape[1]

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    BLOCK_HEADS = 4
    max_heads = max(num_heads_q, num_heads_k)
    grid = (
        seq_len,
        triton.cdiv(max_heads, BLOCK_HEADS),
        2,   # dim2轴分流：0=Q, 1=K
    )

    fused_rope_kernel[grid](
        q, k, out_q, out_k,
        angles,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        angles.stride(0), angles.stride(1),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        HEAD_DIM=head_dim,
        BLOCK_HEADS=BLOCK_HEADS,
    )
    return out_q, out_k

def test_correctness():
    torch.manual_seed(42)
    cases = [
        (256,  32, 8, 128),   # seq, num_heads_q, num_heads_k(GQA机制), head_dim
        (512,  32, 8, 128),
        (1024, 32, 8, 64),
    ]
    print("🚀 开始引爆你的完全体 Fused RoPE 算子...")
    for seq_len, nq, nk, hd in cases:
        q = torch.randn(seq_len, nq, hd, device='cuda', dtype=torch.float16)
        k = torch.randn(seq_len, nk, hd, device='cuda', dtype=torch.float16)
        angles = precompute_freqs(hd, seq_len)  

        cos_val = torch.cos(angles).to(torch.float16)  
        sin_val = torch.sin(angles).to(torch.float16)

        def ref_rope(x):
            x1 = x[..., :hd//2]
            x2 = x[..., hd//2:]
            c = cos_val[:, None, :]   
            s = sin_val[:, None, :]
            return torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)

        ref_q = ref_rope(q)
        ref_k = ref_rope(k)

        out_q, out_k = rope(q, k, angles)

        diff_q = (ref_q - out_q).abs().max().item()
        diff_k = (ref_k - out_k).abs().max().item()
        ok = diff_q < 0.01 and diff_k < 0.01
        print(f"  seq={seq_len:<4} nq={nq} nk={nk} hd={hd:<3}: "
              f"Q_diff={diff_q:.2e} K_diff={diff_k:.2e} "
              f"{'✅' if ok else '❌'}")

if __name__ == "__main__":
    test_correctness()