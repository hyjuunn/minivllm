"""
attention/triton_backend.py
TODO
"""
from minivllm.attention.backend import AttentionBackend, register_backend


@register_backend("triton")
class TritonAttentionBackend(AttentionBackend):
    def forward(self, q, k, v, cache, layer_idx, batch):
        raise NotImplementedError(
            "not implemented"
        )
