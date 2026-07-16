"""
config.py - model/engine setting

ModelConfig : model structure info read from HF config.json
EngineConfig : engine setting from user (dtype, backend..etc)
"""

import json
import os
from dataclasses import dataclass, field
import torch


@dataclass
class ModelConfig:
    architectures: list
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
    max_position: int

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ModelConfig":
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        return cls(
            architectures=c.get("architectures", []),
            hidden_size=c["hidden_size"],
            n_layers=c["num_hidden_layers"],
            n_heads=c["num_attention_heads"],
            n_kv_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
            intermediate_size=c["intermediate_size"],
            vocab_size=c["vocab_size"],
            rms_eps=c.get("rms_norm_eps", 1e-6),
            rope_theta=c.get("rope_theta", 10000.0),
            tie_embeddings=c.get("tie_word_embeddings", False),
            max_position=c.get("max_position_embeddings", 32768),
        )


@dataclass
class EngineConfig:
    model_dir: str
    max_len: int = 4096                  # KV cache max len
    attention_backend: str = "auto"      # "naive" | "sdpa" | "triton" | "auto"
    kv_cache: str = "simple"             # "simple" | "paged"
    dtype: str = "auto"                  # "auto" | "float32" | "bfloat16" | "float16"
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # function to determine dtype based on hardware
    def resolve_dtype(self, device: str) -> torch.dtype:
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        if device == "cuda":
            # rtx 4060 uses bfloat16 as native?
            return torch.bfloat16
        # else
        return torch.float32
