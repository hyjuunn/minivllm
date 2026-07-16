"""
kernels/rmsnorm.py

usage: MINIVLLM_TRITON_RMSNORM=1 python examples/chat.py --model ...
"""
import torch

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x2d = x.contiguous().view(-1, N)
    y = torch.empty_like(x2d)
    BLOCK = triton.next_power_of_2(N)

    _rmsnorm_kernel[(x2d.shape[0],)](
        x2d, weight, y, N, eps, BLOCK=BLOCK,
        num_warps=8 if BLOCK >= 2048 else 4,
    )
    return y.view(orig_shape)
