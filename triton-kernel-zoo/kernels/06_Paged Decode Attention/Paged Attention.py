import triton
import triton.language as tl
import torch

# key_cache:   [num_physical_blocks, num_heads, block_size, head_dim]
# block_table: [batch_size, max_num_blocks_per_seq]
# context_lens:[batch_size]   每个序列已有多少个KV
@triton.jit
def paged_decode_attention_kernel(
    Output_ptr, Q_ptr,
    K_Cache_ptr, V_Cache_ptr,
    Block_table_ptr,
    Context_lens_ptr,
    stride_qm, stride_qh, stride_qd,   # ✅ 精准对齐 3 维 Q 的步长
    stride_kblock, stride_kh, stride_km, stride_kd,
    stride_vblock, stride_vh, stride_vm, stride_vd,
    stride_bt_b, stride_bt_n,
    num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,        # KV cache的block大小（如16）
    BLOCK_D: tl.constexpr,           # head_dim，constexpr版本
):
    # 每个program处理一个(batch, head)对
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # 加载这个序列的 context 长度
    context_len = tl.load(Context_lens_ptr + batch_id)

    # 加载当前的 Query 向量，shape=[HEAD_DIM]
    offs_d = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q_ptr + batch_id * stride_qm + head_id * stride_qh + offs_d * stride_qd
    q = tl.load(Q_ptrs).to(tl.float32)   # [HEAD_DIM]

    # 初始化 online softmax 状态标量 (Decode 阶段 BLOCK_Q = 1)
    m_i = tl.zeros((), dtype=tl.float32) - float("inf")
    l_i = 0.0
    scale = 1.0 / tl.sqrt(float(HEAD_DIM))
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # 计算需要遍历多少个 KV block
    num_blocks = tl.cdiv(context_len, BLOCK_SIZE)

    # 遍历所有物理 KV block
    for block_id in range(0, num_blocks):
        # 从 block_table 里查出物理 block 编号
        physical_block_id = tl.load(Block_table_ptr + batch_id * stride_bt_b + block_id * stride_bt_n)

        block_start = block_id * BLOCK_SIZE
        # ✅ 1. 修正拼写错误 tl.minimum
        block_end = tl.minimum(BLOCK_SIZE, context_len - block_start)

        # 加载这个 block 里所有 K 和 V 向量表
        # ✅ 2. 修正 slot_mask 的一维刚性 Shape，对齐随后的一维 scores
        slots = tl.arange(0, BLOCK_SIZE)         # [BLOCK_SIZE] 
        slot_mask = slots < block_end           # [BLOCK_SIZE] 纯一维布尔长线

        # 构造 K 和 V 的指针矩阵
        K_ptrs = K_Cache_ptr + physical_block_id * stride_kblock + head_id * stride_kh + slots[:, None] * stride_km + offs_d[None, :] * stride_kd
        V_ptrs = V_Cache_ptr + physical_block_id * stride_vblock + head_id * stride_vh + slots[:, None] * stride_vm + offs_d[None, :] * stride_vd

        k = tl.load(K_ptrs, mask=slot_mask[:, None], other=0.0)   # [BLOCK_SIZE, HEAD_DIM]
        v = tl.load(V_ptrs, mask=slot_mask[:, None], other=0.0)   # [BLOCK_SIZE, HEAD_DIM]

        # 计算当前 block 的 attention scores (向量乘以矩阵每行，再沿列求和消消乐)
        # q[None, :] 广播成 [BLOCK_SIZE, HEAD_DIM]，乘以 k 之后沿 axis=1 求和
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_SIZE]

        # 超出当前有效 slot 边界的位置无情填入负无穷大
        scores = tl.where(slot_mask, scores, -float("inf"))

        # Online softmax 更新 (一维标量版)
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)

        p = tl.exp(scores - m_new)        # [BLOCK_SIZE]
        l_new = alpha * l_i + tl.sum(p, axis=0)

        # acc 更新：一维向量级别的内积平移
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # 最终分母归一化
    out = acc / l_i   # [HEAD_DIM]
    
    # 精准定位写回 Output_ptr [batch, head, head_dim]
    Out_ptrs = Output_ptr + batch_id * stride_qm + head_id * stride_qh + offs_d * stride_qd
    tl.store(Out_ptrs, out.to(tl.float16))


def paged_decode_attn(q, k_cache, v_cache, block_table, context_lens):
    batch_size, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    BLOCK_SIZE = k_cache.shape[2]   # k_cache: [num_blocks, heads, block_size, dim]

    grid = (batch_size, num_heads)

    # ✅ 3. 铁血修正点：精准传递三维 Q 的物理步长，阻击后续传参大错位！
    paged_decode_attention_kernel[grid](
        out, q, k_cache, v_cache,
        block_table, context_lens,
        q.stride(0), q.stride(1), q.stride(2),  # 👑 3个步长全给齐！
        k_cache.stride(0), k_cache.stride(1),
        k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1),
        v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        num_heads=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_D=head_dim,
    )
    return out


def test_correctness():
    torch.manual_seed(42)
    batch_size = 4
    num_heads  = 8
    head_dim   = 64
    BLOCK_SIZE = 16
    max_ctx    = 128

    context_lens = torch.randint(1, max_ctx, (batch_size,), device='cuda', dtype=torch.int32)
    max_len = context_lens.max().item()

    num_phys_blocks = batch_size * (max_len // BLOCK_SIZE + 1)
    k_cache = torch.randn(num_phys_blocks, num_heads, BLOCK_SIZE, head_dim, device='cuda', dtype=torch.float16)
    v_cache = torch.randn_like(k_cache)

    max_blocks = max_len // BLOCK_SIZE + 1
    block_table = torch.zeros(batch_size, max_blocks, device='cuda', dtype=torch.int32)
    for b in range(batch_size):
        n = (context_lens[b].item() + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table[b, :n] = torch.arange(b * max_blocks, b * max_blocks + n, device='cuda')

    q = torch.randn(batch_size, num_heads, head_dim, device='cuda', dtype=torch.float16)

    ref_out = torch.zeros_like(q)
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        kv_list, vv_list = [], []
        for blk_idx in range((ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE):
            phys = block_table[b, blk_idx].item()
            kv_list.append(k_cache[phys])  
            vv_list.append(v_cache[phys])
        k_all = torch.cat(kv_list, dim=1)[:, :ctx_len, :]  
        v_all = torch.cat(vv_list, dim=1)[:, :ctx_len, :]
        
        scale = head_dim ** -0.5
        scores = torch.bmm(
            q[b].unsqueeze(1).float(),     
            k_all.float().transpose(1, 2)  
        ).squeeze(1) * scale               
        attn = torch.softmax(scores, dim=-1)
        ref_out[b] = torch.bmm(
            attn.unsqueeze(1),             
            v_all.float()                  
        ).squeeze(1).half()

    out = paged_decode_attn(q, k_cache, v_cache, block_table, context_lens)

    print("Testing Paged Decode Attention...")
    for b in range(batch_size):
        diff = (ref_out[b] - out[b]).abs().max().item()
        print(f"  batch[{b}] ctx={context_lens[b].item():3d}: max_diff={diff:.2e} {'✅' if diff < 0.05 else '❌'}")

if __name__ == "__main__":
    test_correctness()