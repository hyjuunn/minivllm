"""
attention/naive.py - pure Pytorch reference backend

basic implementation
will be the answer sheet for new backends 
new kernel output should be allclose with this
"""
import math

import torch
import torch.nn.functional as F

from minivllm.attention.backend import AttentionBackend, register_backend


@register_backend("naive")
class NaiveAttentionBackend(AttentionBackend):
    def forward(self, q, k, v, cache, layer_idx, batch):
        B, H, T, D = q.shape

        # write cache + get entire past k/v [B, KVH, S, D], S = max_seq_len
        k_all, v_all = cache.update(layer_idx, batch.slots, batch.positions, batch.max_seq_len, k, v)

        # GQA expansion
        k_all, v_all = self.expand_gqa(k_all, v_all, H)

        # softmax(QK^T / sqrt(d)) V
        scores = q @ k_all.transpose(-2, -1) / math.sqrt(D) # [B, H, T, S]
        if T > 1:
            # prefill: causal mask. query i allowed to see key j <= i
            # (general form is start_pos + 1 for chunked prefill (TODO))
            S = scores.shape[-1]
            mask = torch.triu(
                torch.full((T, S), float("-inf"), device=q.device),
                diagonal=1)
            scores = scores + mask
        # decode(T=1)
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        return attn @ v_all # [B, H, T, D]
