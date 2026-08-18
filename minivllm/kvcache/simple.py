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

    def update(self, layer: int, slots, positions, max_seq_len, k, v):
        """
        record k,v [B, KVH, T, D] for this step and return the cache rows 0..max_seq_len
        slot [B] picks the row, and positions [B,T] picks the length offset
        """
        # prevent exceeding limit
        assert max_seq_len <= self.max_len, f"KV cache exceeded: {max_seq_len} > {self.max_len}. Increase --max-len"

        # write: slots - [B,1] so it broadcasts against positions [B,T] and pairs row i with its own positions
        # read: slots- [B]
        self.k[layer][slots.unsqueeze(1), :, positions] = k.transpose(1,2)
        self.v[layer][slots.unsqueeze(1), :, positions] = v.transpose(1,2)
        return (self.k[layer][slots, :, :max_seq_len], self.v[layer][slots, :, :max_seq_len])

    def memory_bytes(self) -> int:
        """memory taken by cache"""
        return self.k.numel() * self.k.element_size() * 2 #*2 since kv
