from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


CUDA_SOURCE = Path(__file__).resolve().parent / "kernels" / "gemm_kernels.cu"


@lru_cache(maxsize=1)
def _extension():
    return load(
        name="profile_driven_cuda_gemm",
        sources=[str(CUDA_SOURCE)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def cuda_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension().gemm_naive(a.contiguous(), b.contiguous())


def cuda_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension().gemm_tiled(a.contiguous(), b.contiguous())


def cuda_reg_blocked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension().gemm_reg_blocked(a.contiguous(), b.contiguous())


def cuda_vec4(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _extension().gemm_vec4(a.contiguous(), b.contiguous())


PROVIDERS = {
    "torch_matmul": torch_matmul,
    "cuda_naive": cuda_naive,
    "cuda_tiled": cuda_tiled,
    "cuda_reg_blocked": cuda_reg_blocked,
    "cuda_vec4": cuda_vec4,
}

