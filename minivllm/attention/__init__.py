"""attention package - importing it triggers backend registration."""
from minivllm.attention.backend import AttentionBackend, get_attention_backend, register_backend
from minivllm.attention import naive
from minivllm.attention import sdpa     
from minivllm.attention import triton_backend

__all__ = ["AttentionBackend", "get_attention_backend", "register_backend"]
