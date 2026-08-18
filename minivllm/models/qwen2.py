"""
models/qwen2.py - Qwen2 architecture implementation

how to add new model
 - copy this file and modify arch (ex) if moe, swiglumlp -> moe block)
 - match param name to HF checkpoint
 - register to registry.py XxxForCausalLM
 - use tests/test_correctness.py to compare logits with transformers 
"""
import torch
import torch.nn as nn

from minivllm.config import ModelConfig
from minivllm.engine.forward_batch import ForwardBatch
from minivllm.models.layers import RMSNorm, RotaryEmbedding, SwiGLUMLP


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, backend, rope: RotaryEmbedding):
        super().__init__()
        self.layer_idx = layer_idx
        self.backend = backend
        self.rope = rope
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        h, d = cfg.hidden_size, cfg.head_dim

        # qwen2 has bias on q, k, v but not on o_proj
        self.q_proj = nn.Linear(h, cfg.n_heads * d, bias=True)
        self.k_proj = nn.Linear(h, cfg.n_kv_heads * d, bias=True)
        self.v_proj = nn.Linear(h, cfg.n_kv_heads * d, bias=True)
        self.o_proj = nn.Linear(cfg.n_heads * d, h, bias=False)

    def forward(self, x, batch: ForwardBatch, cache):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, batch.positions)
        out = self.backend.forward(q, k, v, cache, self.layer_idx, batch)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx, backend, rope):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.self_attn = Qwen2Attention(cfg, layer_idx, backend, rope)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, batch: ForwardBatch, cache):
        # pre-norm residual structure
        x = x + self.self_attn(self.input_layernorm(x), batch, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg, backend, max_len: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        rope = RotaryEmbedding(cfg.head_dim, max_len, cfg.rope_theta)
        # self.rope = rope to auto move onto gpu
        self.rope = rope
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(cfg, i, backend, rope) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)

    def forward(self, input_ids, batch: ForwardBatch, cache):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, batch, cache)
        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, backend, max_len: int):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg, backend, max_len)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, batch: ForwardBatch, cache) -> torch.Tensor:
        """input_ids [B, T] -> hidden states [B, T, hidden]"""
        return self.model(input_ids, batch, cache)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [..., hidden] -> logits [..., vocab]"""
        return self.lm_head(hidden)
