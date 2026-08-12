"""
tests/test_correctness.py - compare logit with hf reference.

run:
  export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
  pytest tests/test_correctness.py -v
"""
import os

import pytest
import torch

from minivllm.engine.forward_batch import ForwardBatch

MODEL_DIR = os.environ.get("MINIVLLM_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    MODEL_DIR is None, reason="MINIVLLM_TEST_MODEL=<model path> setup needed")

PROMPT = "The quick brown fox"


def _ref_logits(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).to(device).eval()
    with torch.inference_mode():
        return ids, model(ids).logits


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_prefill_logits_match(backend):
    from minivllm import EngineConfig, LLMEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, ref = _ref_logits(device)

    engine = LLMEngine(EngineConfig(
        model_dir=MODEL_DIR, dtype="float32", attention_backend=backend,
        max_len=ids.shape[1] + 8))
    cache = engine._new_cache()
    with torch.inference_mode():
        batch = ForwardBatch.for_prefill(ids.shape[1], slot=0, device=device)
        mine = engine.model.compute_logits(engine.model(ids, batch, cache))

    diff = (ref - mine).abs().max().item()
    assert diff < 1e-3, (
        f"[{backend}] logits max diff={diff:.5f}\n"
        )


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_greedy_decode_matches_hf(backend):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from minivllm import EngineConfig, LLMEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenize
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]

    # answer + reference scores
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32
    ).to(device).eval()
    with torch.inference_mode():
        gen = ref_model.generate(ids, max_new_tokens=10, do_sample=False)
        ref_logits = ref_model(gen).logits[0]
    seq = gen[0].tolist()

    # build engine and prefill prompt
    engine = LLMEngine(EngineConfig(
        model_dir=MODEL_DIR, dtype="float32", attention_backend=backend, max_len=256
    ))
    cache = engine._new_cache()
    with torch.inference_mode():
        pre = engine.model(
            ids, ForwardBatch.for_prefill(prompt_len, slot=0, device=device), cache)

        for p in range(prompt_len, prompt_len + 10):
            if p == prompt_len:
                mine = engine.model.compute_logits(pre[0, -1])
            else:
                step = torch.tensor([[seq[p - 1]]], device=device)
                batch = ForwardBatch.for_decode([p - 1], slots=[0], device=device)
                hidden = engine.model(step, batch, cache)
                mine = engine.model.compute_logits(hidden[0,-1])
            ref = ref_logits[p-1]
            diff = (mine - ref).abs().max().item()
            assert diff < 1e-3, f"[{backend}] step {p - prompt_len}: logit diff {diff:.2e}"

