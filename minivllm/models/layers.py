"""
models/layers.py — basic layers shared by models

reuse these layers when adding new model
RMSNorm can be replaced by Triton kernel (see kernels/rmsnorm.py) 
(example to show how kernel optimization connects to model code)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _triton_rmsnorm_enabled(x: torch.Tensor) -> bool:
    """use env var to on/off Triton RMSNorm"""
    return (
        os.environ.get("MINIVLLM_TRITON_RMSNORM", "0") == "1"
        and x.is_cuda
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _triton_rmsnorm_enabled(x):
            from minivllm.kernels.rmsnorm import rmsnorm_triton
            return rmsnorm_triton(x, self.weight, self.eps)
        # reference implementation (use fp32)
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """
    rope cos/sin table
    not a parameter -> register as non-persistent buffer
    load_state_dict doesn't complain even if not in checkpoint
    """

    def __init__(self, head_dim: int, max_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq) # [max_len, head_dim/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q, k, positions: torch.Tensor):
        """
        q: [B, H, T, D], k: [B, KVH, T, D], positions: [B, T] -> (q, k) with rope applied

        gather, not slice: a decode batch holds sequences at unrelated positions
        (37, 512, 8, ...) which no contiguous start:stop can express
        """
        cos = self.cos[positions].to(q.dtype).unsqueeze(1) #[B,T,D/2] -> [B,1,T,D/2]
        sin = self.sin[positions].to(q.dtype).unsqueeze(1)
        # head dim stays 1 -> broadcasts over heads in _rotate
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLUMLP(nn.Module):
    """
    down(silu(gate(x)) * up(x))
    for MoE model, make multiple classes and use as expert
    and put router at front
    """ 

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
