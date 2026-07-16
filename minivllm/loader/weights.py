"""loader/weights.py — safetensors -> model load

Memory Tip: Using load_state_dict(assign=True) 'swaps' the initialized 
parameters with checkpoint tensors (instead of copying). 
To take it a step further, the vLLM approach creates the model on the meta 
device to eliminate initial memory allocation entirely
"""
import glob
import os

import torch
from safetensors.torch import load_file

from minivllm.config import EngineConfig, ModelConfig
from minivllm.models.registry import get_model_class


def load_model(model_cfg: ModelConfig, engine_cfg: EngineConfig, backend,
               device: str, dtype: torch.dtype):
    model_cls = get_model_class(model_cfg.architectures)

    # generate model skeleton
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = model_cls(model_cfg, backend, engine_cfg.max_len)
    finally:
        torch.set_default_dtype(prev_dtype)

    # checkpoint tensor load
    files = sorted(glob.glob(os.path.join(engine_cfg.model_dir, "*.safetensors")))
    assert files, f"no .safetensors in {engine_cfg.model_dir}"
    state = {}
    for f in files:
        for name, t in load_file(f).items():
            state[name] = t.to(device=device, dtype=dtype)

    # tied embeddings
    if model_cfg.tie_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    return model
