from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(
    out_ptr,
    gate_ptr,
    up_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = (gate / (1.0 + tl.exp(-gate))) * up

    tl.store(out_ptr + offsets, result, mask=mask)


def pytorch_unfused(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def correctness_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate.float()) * up.float()


def triton_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"gate and up shapes must match, got {gate.shape} and {up.shape}")
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("triton_fused requires CUDA tensors")
    if not gate.is_contiguous() or not up.is_contiguous():
        gate = gate.contiguous()
        up = up.contiguous()

    out = torch.empty_like(gate)
    n_elements = gate.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    _silu_mul_kernel[grid](out, gate, up, n_elements, BLOCK_SIZE=block_size)
    return out


@lru_cache(maxsize=8)
def _compiled_silu_mul(dynamic: bool = False):
    return torch.compile(lambda gate, up: F.silu(gate) * up, dynamic=dynamic)


def torch_compile_fused(
    gate: torch.Tensor,
    up: torch.Tensor,
    dynamic: bool = False,
) -> torch.Tensor:
    return _compiled_silu_mul(dynamic)(gate, up)


PROVIDERS = {
    "pytorch_unfused": pytorch_unfused,
    "torch_compile": torch_compile_fused,
    "triton": triton_fused,
}

