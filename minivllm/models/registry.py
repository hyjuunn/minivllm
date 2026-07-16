"""
models/registry.py - maps the architectures name from config.json to an implementation class

Plays the same role as vLLM's vllm/model_executor/models/registry.py.
To support a new model, just add one line here
"""
from minivllm.models.qwen2 import Qwen2ForCausalLM

MODEL_REGISTRY = {
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    # "Qwen3ForCausalLM": ...,      # practice: Qwen3 with q/k norm added
    # "LlamaForCausalLM": ...,      # practice: Llama with no bias
    # "Qwen2MoeForCausalLM": ...,   # Moe
}


def get_model_class(architectures: list):
    for arch in architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(
        f"Not supported architecture {architectures}. "
        f"Supported list: {list(MODEL_REGISTRY)} — add to models/ and register")
