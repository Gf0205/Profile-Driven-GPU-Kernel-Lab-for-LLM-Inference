import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_prefill_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    stride_qm, stride_qh, stride_qd,   
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd, 
    stride_om, stride_oh, stride_od,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    head_idx = tl.program_id(0)
    block_q_id = tl.program_id(1)
    
    # 1. 构造 Q 维度的指针大盘 [BLOCK_Q, HEAD_DIM]
    q_start_ptr = Q_ptr + head_idx * stride_qh + block_q_id * BLOCK_Q * stride_qm
    offs_q = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_start_ptr + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qd
    mask = (offs_q[:, None] + block_q_id * BLOCK_Q < seq_len)
    q = tl.load(q_ptrs, mask=mask, other=0.0)     

    # 初始化 online softmax 状态标量 (保持 fp32 进行高精度中间累加)
    m_i = tl.zeros((BLOCK_Q,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(float(HEAD_DIM)) 

    # Causal 模式下只需遍历到当前 query block 的结束边界
    kv_end = tl.cdiv((block_q_id * BLOCK_Q + BLOCK_Q), BLOCK_KV) if CAUSAL else tl.cdiv(seq_len, BLOCK_KV)

    for block_kv_id in range(0, kv_end):
        start_kv = block_kv_id * BLOCK_KV
        offs_k = tl.arange(0, BLOCK_KV)
        
        # 👑 算出当前 Key 和 Query 在全场大盘里的【绝对全局物理工号】
        global_q_idx = block_q_id * BLOCK_Q + offs_q[:, None]
        global_k_idx = block_kv_id * BLOCK_KV + offs_k[None, :]

        # 2. K 指针矩阵化布局 [BLOCK_KV, HEAD_DIM]
        k_ptrs = (
            K_ptr 
            + head_idx * stride_kh 
            + (start_kv + offs_k[:, None]) * stride_km 
            + offs_d[None, :] * stride_kd
        )
        k_mask = (start_kv + offs_k[:, None]) < seq_len
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)      

        # 3. 👑 V 指针矩阵化布局 [HEAD_DIM, BLOCK_KV] (完美对齐长线维度的 Stride)
        v_ptrs = (
            V_ptr 
            + head_idx * stride_vh 
            + offs_d[:, None] * stride_vd 
            + (start_kv + offs_k[None, :]) * stride_vm
        )
        v_mask = (start_kv + offs_k[None, :]) < seq_len
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)      

        # 计算 attention scores：Q @ K^T -> [BLOCK_Q, BLOCK_KV]
        scores = tl.dot(q, k.trans()) * scale    

        # 👑 铁血修正点：使用全局绝对坐标，给局部的 scores 矩阵上最完美的刚性锁！
        if CAUSAL:
            causal_mask = (global_q_idx >= global_k_idx)
            scores = tl.where(causal_mask, scores, -1.0e9)

        # 边界保护：防止序列末尾未对齐部分的垃圾残渣参与 Softmax 运算
        scores_mask = global_k_idx < seq_len
        scores = tl.where(scores_mask, scores, -1.0e9)

        # Online Softmax 核心更新三连
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # 状态平移累加
        acc = alpha[:, None] * acc
        # 💥 p 强转 fp16，配合 v.trans() [BLOCK_KV, HEAD_DIM] 完美引爆 Tensor Core 机器码！
        acc = acc + tl.dot(p.to(tl.float16), v.trans())

        m_i = m_new
        l_i = l_new

    # 最终分母归一化
    acc = acc / l_i[:, None]

    # 二维并行写回全局显存 O
    out_ptrs = Output_ptr + head_idx * stride_oh + (block_q_id * BLOCK_Q + offs_q[:, None]) * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_q[:, None] + block_q_id * BLOCK_Q < seq_len)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)

def flash_attn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    seq_len, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    BLOCK_Q = 64
    BLOCK_KV = 64

    num_query_blocks = triton.cdiv(seq_len, BLOCK_Q)
    grid = (num_heads, num_query_blocks)
    
    flash_attn_prefill_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), 
        out.stride(0), out.stride(1), out.stride(2),
        seq_len,
        HEAD_DIM=head_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        CAUSAL=causal,
    )
    return out

def test_correctness():
    torch.manual_seed(42)
    cases = [
        (256,  8, 64),
        (512,  8, 64),
    ]
    print("🚀 开始引爆你的完全体 Triton FlashAttention 算子...")
    for seq_len, num_heads, head_dim in cases:
        q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        scale = head_dim ** -0.5
        q_t = q.transpose(0, 1).float()
        k_t = k.transpose(0, 1).float()
        v_t = v.transpose(0, 1).float()
        
        scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale
        mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        ref = torch.bmm(attn, v_t).transpose(0, 1).half()

        out = flash_attn_prefill(q, k, v, causal=True)

        # 允许半精度浮点数在海量累加中存在万分级（1e-3）的正常硬件舍入气泡
        max_diff = (ref - out).abs().max().item()
        print(f"  seq={seq_len:4d} h={num_heads} d={head_dim}: max_diff={max_diff:.2e} {'✅' if max_diff < 0.01 else '❌'}")

if __name__ == "__main__":
    test_correctness()