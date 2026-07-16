"""
kvcache/simple.py - simple contiguous KV cache

TODO (solve with paged kv cache)
- allocates the full max_len per sequence -> wasting memory for short req
- requires contiguous memory for sequences in a batch -> unmanagable with variable lengths
- cannot share cache across identical prompts - no prefix caching
"""

import torch


class SimpleKVCache:
    def __init__(self, n_layers: int, batch: int, n_kv_heads: int, max_len: int, head_dim: int, dtype, device):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len

    def update(self, layer: int, start_pos: int, k, v):
        """record k,v [B, KVH, T, D] for this step and return entire 0..start_pos+T"""
        T = k.shape[2]
        # prevent exceeding limit
        assert start_pos + T <= self.max_len, f"KV cache Exceeded: {start_pos + T} > {self.max_len}. Increase --max-len"
        self.k[layer][:, :, start_pos:start_pos + T] = k
        self.v[layer][:, :, start_pos:start_pos + T] = v
        return (self.k[layer][:, :, : start_pos + T], self.v[layer][:, :, : start_pos + T])

    def memory_bytes(self) -> int:
        """memory taken by cache"""
        return self.k.numel() * self.k.element_size() * 2 #*2 since kv
