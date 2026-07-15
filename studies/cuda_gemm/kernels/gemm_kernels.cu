#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

namespace {

constexpr int TILE = 16;
constexpr int RB_TILE = 16;
constexpr int WMMA_TILE = 16;
constexpr int WMMA_BLOCK_M = 64;
constexpr int WMMA_BLOCK_N = 32;
constexpr int WMMA_BLOCK_WARPS = 8;
constexpr int WARP_SIZE = 32;

struct alignas(8) Half4 {
    half x;
    half y;
    half z;
    half w;
};

__global__ void gemm_naive_kernel(const half* __restrict__ a,
                                  const half* __restrict__ b,
                                  half* __restrict__ c,
                                  int m,
                                  int n,
                                  int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
    }
    c[row * n + col] = __float2half(acc);
}

__global__ void gemm_tiled_kernel(const half* __restrict__ a,
                                  const half* __restrict__ b,
                                  half* __restrict__ c,
                                  int m,
                                  int n,
                                  int k) {
    __shared__ half as[TILE][TILE];
    __shared__ half bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int tile_k = 0; tile_k < k; tile_k += TILE) {
        int a_col = tile_k + tx;
        int b_row = tile_k + ty;
        as[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : __float2half(0.0f);
        bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : __float2half(0.0f);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += __half2float(as[ty][kk]) * __half2float(bs[kk][tx]);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = __float2half(acc);
    }
}

__global__ void gemm_reg_blocked_kernel(const half* __restrict__ a,
                                        const half* __restrict__ b,
                                        half* __restrict__ c,
                                        int m,
                                        int n,
                                        int k) {
    __shared__ half as[RB_TILE][RB_TILE];
    __shared__ half bs[RB_TILE][RB_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.y * RB_TILE + ty * 2;
    int col0 = blockIdx.x * RB_TILE + tx * 2;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (int tile_k = 0; tile_k < k; tile_k += RB_TILE) {
        int a_col0 = tile_k + tx * 2;
        int b_row0 = tile_k + ty * 2;

        as[ty * 2][tx * 2] = (row0 < m && a_col0 < k) ? a[row0 * k + a_col0] : __float2half(0.0f);
        as[ty * 2][tx * 2 + 1] = (row0 < m && a_col0 + 1 < k) ? a[row0 * k + a_col0 + 1] : __float2half(0.0f);
        as[ty * 2 + 1][tx * 2] = (row0 + 1 < m && a_col0 < k) ? a[(row0 + 1) * k + a_col0] : __float2half(0.0f);
        as[ty * 2 + 1][tx * 2 + 1] = (row0 + 1 < m && a_col0 + 1 < k) ? a[(row0 + 1) * k + a_col0 + 1] : __float2half(0.0f);

        bs[ty * 2][tx * 2] = (b_row0 < k && col0 < n) ? b[b_row0 * n + col0] : __float2half(0.0f);
        bs[ty * 2][tx * 2 + 1] = (b_row0 < k && col0 + 1 < n) ? b[b_row0 * n + col0 + 1] : __float2half(0.0f);
        bs[ty * 2 + 1][tx * 2] = (b_row0 + 1 < k && col0 < n) ? b[(b_row0 + 1) * n + col0] : __float2half(0.0f);
        bs[ty * 2 + 1][tx * 2 + 1] = (b_row0 + 1 < k && col0 + 1 < n) ? b[(b_row0 + 1) * n + col0 + 1] : __float2half(0.0f);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < RB_TILE; ++kk) {
            float a0 = __half2float(as[ty * 2][kk]);
            float a1 = __half2float(as[ty * 2 + 1][kk]);
            float b0 = __half2float(bs[kk][tx * 2]);
            float b1 = __half2float(bs[kk][tx * 2 + 1]);
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < m && col0 < n) {
        c[row0 * n + col0] = __float2half(acc00);
    }
    if (row0 < m && col0 + 1 < n) {
        c[row0 * n + col0 + 1] = __float2half(acc01);
    }
    if (row0 + 1 < m && col0 < n) {
        c[(row0 + 1) * n + col0] = __float2half(acc10);
    }
    if (row0 + 1 < m && col0 + 1 < n) {
        c[(row0 + 1) * n + col0 + 1] = __float2half(acc11);
    }
}

__global__ void gemm_vec4_kernel(const half* __restrict__ a,
                                 const half* __restrict__ b,
                                 half* __restrict__ c,
                                 int m,
                                 int n,
                                 int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    int kk = 0;
    for (; kk + 3 < k; kk += 4) {
        Half4 av = *reinterpret_cast<const Half4*>(&a[row * k + kk]);
        acc += __half2float(av.x) * __half2float(b[(kk + 0) * n + col]);
        acc += __half2float(av.y) * __half2float(b[(kk + 1) * n + col]);
        acc += __half2float(av.z) * __half2float(b[(kk + 2) * n + col]);
        acc += __half2float(av.w) * __half2float(b[(kk + 3) * n + col]);
    }
    for (; kk < k; ++kk) {
        acc += __half2float(a[row * k + kk]) * __half2float(b[kk * n + col]);
    }
    c[row * n + col] = __float2half(acc);
}

__global__ void gemm_wmma_kernel(const half* __restrict__ a,
                                 const half* __restrict__ b,
                                 half* __restrict__ c,
                                 int m,
                                 int n,
                                 int k) {
    using namespace nvcuda;

    __shared__ half a_tile[WMMA_TILE * WMMA_TILE];
    __shared__ half b_tile[WMMA_TILE * WMMA_TILE];
    __shared__ float c_tile[WMMA_TILE * WMMA_TILE];

    int tile_m = blockIdx.y * WMMA_TILE;
    int tile_n = blockIdx.x * WMMA_TILE;
    int lane = threadIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += WMMA_TILE) {
        for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += 32) {
            int row = idx / WMMA_TILE;
            int col = idx % WMMA_TILE;
            int global_a_row = tile_m + row;
            int global_a_col = tile_k + col;
            int global_b_row = tile_k + row;
            int global_b_col = tile_n + col;

            a_tile[idx] = (global_a_row < m && global_a_col < k)
                              ? a[global_a_row * k + global_a_col]
                              : __float2half(0.0f);
            b_tile[idx] = (global_b_row < k && global_b_col < n)
                              ? b[global_b_row * n + global_b_col]
                              : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_tile, WMMA_TILE);
        wmma::load_matrix_sync(b_frag, b_tile, WMMA_TILE);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_tile, acc_frag, WMMA_TILE, wmma::mem_row_major);
    __syncthreads();

    for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += 32) {
        int row = idx / WMMA_TILE;
        int col = idx % WMMA_TILE;
        int global_row = tile_m + row;
        int global_col = tile_n + col;
        if (global_row < m && global_col < n) {
            c[global_row * n + global_col] = __float2half(c_tile[idx]);
        }
    }
}

__global__ void gemm_wmma_block_tiled_kernel(const half* __restrict__ a,
                                             const half* __restrict__ b,
                                             half* __restrict__ c,
                                             int m,
                                             int n,
                                             int k) {
    using namespace nvcuda;

    __shared__ half a_tiles[WMMA_BLOCK_WARPS][WMMA_TILE * WMMA_TILE];
    __shared__ half b_tiles[WMMA_BLOCK_WARPS][WMMA_TILE * WMMA_TILE];
    __shared__ float c_tiles[WMMA_BLOCK_WARPS][WMMA_TILE * WMMA_TILE];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;
    int tile_m = blockIdx.y * WMMA_BLOCK_M + warp_m * WMMA_TILE;
    int tile_n = blockIdx.x * WMMA_BLOCK_N + warp_n * WMMA_TILE;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += WMMA_TILE) {
        for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += WARP_SIZE) {
            int row = idx / WMMA_TILE;
            int col = idx % WMMA_TILE;
            int global_a_row = tile_m + row;
            int global_a_col = tile_k + col;
            int global_b_row = tile_k + row;
            int global_b_col = tile_n + col;

            a_tiles[warp_id][idx] = (global_a_row < m && global_a_col < k)
                                        ? a[global_a_row * k + global_a_col]
                                        : __float2half(0.0f);
            b_tiles[warp_id][idx] = (global_b_row < k && global_b_col < n)
                                        ? b[global_b_row * n + global_b_col]
                                        : __float2half(0.0f);
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, a_tiles[warp_id], WMMA_TILE);
        wmma::load_matrix_sync(b_frag, b_tiles[warp_id], WMMA_TILE);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c_tiles[warp_id], acc_frag, WMMA_TILE, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += WARP_SIZE) {
        int row = idx / WMMA_TILE;
        int col = idx % WMMA_TILE;
        int global_row = tile_m + row;
        int global_col = tile_n + col;
        if (global_row < m && global_col < n) {
            c[global_row * n + global_col] = __float2half(c_tiles[warp_id][idx]);
        }
    }
}

__global__ void gemm_wmma_shared_tiles_kernel(const half* __restrict__ a,
                                              const half* __restrict__ b,
                                              half* __restrict__ c,
                                              int m,
                                              int n,
                                              int k) {
    using namespace nvcuda;

    constexpr int WARP_M_TILES = WMMA_BLOCK_M / WMMA_TILE;
    constexpr int WARP_N_TILES = WMMA_BLOCK_N / WMMA_TILE;
    __shared__ half a_tiles[WARP_M_TILES][WMMA_TILE * WMMA_TILE];
    __shared__ half b_tiles[WARP_N_TILES][WMMA_TILE * WMMA_TILE];
    __shared__ float c_tiles[WMMA_BLOCK_WARPS][WMMA_TILE * WMMA_TILE];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int warp_m = warp_id / WARP_N_TILES;
    int warp_n = warp_id % WARP_N_TILES;
    int tile_m = blockIdx.y * WMMA_BLOCK_M + warp_m * WMMA_TILE;
    int tile_n = blockIdx.x * WMMA_BLOCK_N + warp_n * WMMA_TILE;

    wmma::fragment<wmma::matrix_a, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE, WMMA_TILE, WMMA_TILE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE, WMMA_TILE, WMMA_TILE, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += WMMA_TILE) {
        // One warp loads each unique A or B tile; all consumer warps reuse it.
        if (warp_n == 0) {
            for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += WARP_SIZE) {
                int row = idx / WMMA_TILE;
                int col = idx % WMMA_TILE;
                int global_row = tile_m + row;
                int global_col = tile_k + col;
                a_tiles[warp_m][idx] = (global_row < m && global_col < k)
                                                ? a[global_row * k + global_col]
                                                : __float2half(0.0f);
            }
        }
        if (warp_m == 0) {
            for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += WARP_SIZE) {
                int row = idx / WMMA_TILE;
                int col = idx % WMMA_TILE;
                int global_row = tile_k + row;
                int global_col = tile_n + col;
                b_tiles[warp_n][idx] = (global_row < k && global_col < n)
                                                ? b[global_row * n + global_col]
                                                : __float2half(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, a_tiles[warp_m], WMMA_TILE);
        wmma::load_matrix_sync(b_frag, b_tiles[warp_n], WMMA_TILE);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(c_tiles[warp_id], acc_frag, WMMA_TILE, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < WMMA_TILE * WMMA_TILE; idx += WARP_SIZE) {
        int row = idx / WMMA_TILE;
        int col = idx % WMMA_TILE;
        int global_row = tile_m + row;
        int global_col = tile_n + col;
        if (global_row < m && global_col < n) {
            c[global_row * n + global_col] = __float2half(c_tiles[warp_id][idx]);
        }
    }
}

torch::Tensor allocate_output(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16, "Only float16 is supported.");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be rank-2 tensors.");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner GEMM dimensions must match.");
    return torch::empty({a.size(0), b.size(1)}, a.options());
}

}  // namespace

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(16, 16);
    dim3 grid((b.size(1) + block.x - 1) / block.x, (a.size(0) + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_tiled(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(TILE, TILE);
    dim3 grid((b.size(1) + TILE - 1) / TILE, (a.size(0) + TILE - 1) / TILE);
    gemm_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_reg_blocked(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(RB_TILE / 2, RB_TILE / 2);
    dim3 grid((b.size(1) + RB_TILE - 1) / RB_TILE, (a.size(0) + RB_TILE - 1) / RB_TILE);
    gemm_reg_blocked_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_vec4(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(16, 16);
    dim3 grid((b.size(1) + block.x - 1) / block.x, (a.size(0) + block.y - 1) / block.y);
    gemm_vec4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_wmma(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(32);
    dim3 grid((b.size(1) + WMMA_TILE - 1) / WMMA_TILE, (a.size(0) + WMMA_TILE - 1) / WMMA_TILE);
    gemm_wmma_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_wmma_block_tiled(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(WMMA_BLOCK_WARPS * WARP_SIZE);
    dim3 grid((b.size(1) + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N, (a.size(0) + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M);
    gemm_wmma_block_tiled_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

torch::Tensor gemm_wmma_shared_tiles(torch::Tensor a, torch::Tensor b) {
    auto c = allocate_output(a, b);
    dim3 block(WMMA_BLOCK_WARPS * WARP_SIZE);
    dim3 grid((b.size(1) + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N, (a.size(0) + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M);
    gemm_wmma_shared_tiles_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        a.size(0),
        b.size(1),
        a.size(1));
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive", &gemm_naive, "Naive FP16 GEMM");
    m.def("gemm_tiled", &gemm_tiled, "Shared-memory tiled FP16 GEMM");
    m.def("gemm_reg_blocked", &gemm_reg_blocked, "Register-blocked FP16 GEMM");
    m.def("gemm_vec4", &gemm_vec4, "Vectorized-load FP16 GEMM");
    m.def("gemm_wmma", &gemm_wmma, "WMMA Tensor Core FP16 GEMM");
    m.def("gemm_wmma_block_tiled", &gemm_wmma_block_tiled, "Block-tiled WMMA Tensor Core FP16 GEMM");
    m.def("gemm_wmma_shared_tiles", &gemm_wmma_shared_tiles, "CTA-shared tile WMMA Tensor Core FP16 GEMM");
}
