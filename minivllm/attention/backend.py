"""
attention/backend.py — Attention Backend interface + registry

This file will be the extension point of this project

Optimizations in vllm boils down to something like "which attention backend/kernel
is selected for this hardware, and how fast is it?" 

vLLM swaps between FlashAttention, Triton, AITER, etc.. using this backend 
abstraction

Trying to build identical structure here:

    - AttentionBackend: The core interface (responsible for writing to the KV cache
    and computing attention)
    - @register_backend("name") : decorator to register new backend implementation
    - MINIVLLM_ATTENTION_BACKEND=naive|sdpa|triton: Env var for switching backends

How to add a new backend: 
    1. Create 'my_backend.py' in this directory and inherit from 'AttentionBackend'
    2. Add the '@register_backend("mybackend")' decorator to the class
    3. Add a single import statement in '__init__.py"
"""

import os
from abc import ABC, abstractmethod

import torch

_BACKENDS: dict[str, type] = {}


def register_backend(name: str):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def get_attention_backend(name: str = "auto") -> "AttentionBackend":
    # order: env > engineconfig > auto
    name = os.environ.get("MINIVLLM_ATTENTION_BACKEND", name)
    if name == "auto":
        name = "sdpa"  # auto logic can be improved
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKENDS)}")
    return _BACKENDS[name]()


class AttentionBackend(ABC):
    """
    Write new K/V into the KV cache -> Compute attention over the entire K/V

    Similar to vLLM, cache writing (e.g., the `reshape_and_cache` kernel) and attention computation
    are handled within the backend domain. When migrating to a paged cache later,
    block_table will be added.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,        # [B, n_heads,    T, head_dim]
        k: torch.Tensor,        # [B, n_kv_heads, T, head_dim]
        v: torch.Tensor,        # [B, n_kv_heads, T, head_dim]
        cache,                  # KVCache obj
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:          # [B, n_heads, T, head_dim]
        ...

    @staticmethod
    def expand_gqa(k_all: torch.Tensor, v_all: torch.Tensor, n_heads: int):
        """GQA: replicate kv head based on q head #"""
        rep = n_heads // k_all.shape[1]
        if rep == 1:
            return k_all, v_all
        return (k_all.repeat_interleave(rep, dim=1),
                v_all.repeat_interleave(rep, dim=1))
