"""
attention/sdpa.py - PyTorch SDPA(scaled_dot_product_attention) backend

F.scaled_dot_product_attention internally selects FlashAttention/memory efficient kernel
automatically
prefill speeds up compared to naive
"""
import torch.nn.functional as F

from minivllm.attention.backend import AttentionBackend, register_backend


@register_backend("sdpa")
class SDPAAttentionBackend(AttentionBackend):
    def forward(self, q, k, v, cache, layer_idx, batch):
        B, H, T, D = q.shape
        k_all, v_all = cache.update(layer_idx, batch.slots, batch.positions, batch.max_seq_len, k, v)
        k_all, v_all = self.expand_gqa(k_all, v_all, H)

        if T > 1:
            # prefill. is_causal=True assumes Q and K starts same
            # TODO: cannot be used for max_seq_len > T (chunked prefill)
            assert batch.max_seq_len == T, "chunked prefill not supported"
            return F.scaled_dot_product_attention(q, k_all, v_all, is_causal=True)

        # decode(T=1)
        return F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=batch.padding_mask)
