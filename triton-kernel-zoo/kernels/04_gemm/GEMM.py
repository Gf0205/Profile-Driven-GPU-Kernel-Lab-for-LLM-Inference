import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr,b_ptr,c_ptr,
    M,N,K,
    stride_am,stride_ak,
    stride_bk,stride_bn,
    stride_cm,stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
  pid = tl.program_id(0)

  #获取每个title上按照BLOCK_N划分的话能划分多少列
  num_pid_n = tl.cdiv(N,BLOCK_N)
  #获取具体的二维小block的行列号
  pid_m = pid // num_pid_n
  pid_n = pid % num_pid_n

  #获取A，B各个tile的各自偏移值
  offs_m = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
  offs_n = pid_n * BLOCK_N + tl.arange(0,BLOCK_N)
  offs_k = tl.arange(0,BLOCK_K)

  #构造A，B的指针

  #从一维指针构建这个tile的二维指针对于A就是[BLOCK_M,BLOCK_N]
  a_ptrs = a_ptr + offs_m[:,None] * stride_am + offs_k[None,:] * stride_ak 
  b_ptrs = b_ptr + offs_k[:,None] * stride_bk + offs_n[None,:] * stride_bn

  acculator = tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
  num_stages = tl.cdiv(K, BLOCK_K)

  #K方向循环
  for stage in range(0,num_stages):
    #A小块的mask掩码部分
    k = stage * BLOCK_K
    # 🚨 终极 2D 动态掩码大闸：横纵双向死守大盘边界
    mask_A = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
    mask_B = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
    #加载A,B各自的数据到片上
    a = tl.load(a_ptrs,mask=mask_A,other=0.0)
    b = tl.load(b_ptrs,mask=mask_B,other=0.0)
    #矩阵乘法累加
    acculator = tl.dot(a,b,acculator)

    #指针沿着K方向不断递进
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

  #处理结束后以float16存储返回

  c = acculator.to(tl.float16)

  #将得到的c结果写到输出指针c_ptr上
  offs_cm = pid_m * BLOCK_M + tl.arange(0,BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0,BLOCK_N)


  #结果c上的小分片tile
  c_ptrs = c_ptr + offs_cm[:,None] * stride_cm + offs_cn[None,:] * stride_cn

  mask_C = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
  tl.store(c_ptrs, c, mask=mask_C)


def matmul(a:torch.Tensor,b:torch.Tensor) -> torch.Tensor:
  M,K = a.shape
  K1,N = b.shape

  assert K1 == K
  c = torch.empty((M,N),device=a.device,dtype=torch.float16)

  BLOCK_M,BLOCK_N,BLOCK_K = 64,64,32
  grid = (triton.cdiv(M,BLOCK_M) * triton.cdiv(N,BLOCK_N),)
  matmul_kernel[grid](
      a,b,c,
      M,N,K,
      a.stride(0), a.stride(1),
      b.stride(0), b.stride(1),
      c.stride(0), c.stride(1),
      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
  )

  return c

def test_correctness():
    torch.manual_seed(42)
    shapes = [
        (64, 64, 64),
        (128, 256, 64),
        (512, 1024, 512),
        (1, 4096, 4096),
        (128, 4096, 4096),
    ]
    print("Testing GEMM correctness...")
    for M, N, K in shapes:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        ref = torch.matmul(a, b)
        out = matmul(a, b)
        max_diff = (ref - out).abs().max().item()
        rel_diff = max_diff / ref.abs().max().item()
        print(f"  ({M:4d},{N:4d},{K:4d}): "
              f"max_diff={max_diff:.2e} rel={rel_diff:.2e} "
              f"{'OK' if rel_diff < 0.01 else 'FAIL'}")

if __name__ == "__main__":
    test_correctness()



