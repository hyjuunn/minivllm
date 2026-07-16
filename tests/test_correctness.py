"""
tests/test_correctness.py - compare logit with hf reference.

run:
  export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
  pytest tests/test_correctness.py -v
"""
import os

import pytest
import torch

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
        mine = engine.model(ids, start_pos=0, cache=cache)

    diff = (ref - mine).abs().max().item()
    assert diff < 1e-3, (
        f"[{backend}] logits max diff={diff:.5f}\n"
        )


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_greedy_decode_matches_hf(backend):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from minivllm import EngineConfig, LLMEngine, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32).to(device).eval()
    with torch.inference_mode():
        ref_out = ref_model.generate(
            ids, max_new_tokens=10, do_sample=False)[0, ids.shape[1]:].tolist()

    engine = LLMEngine(EngineConfig(
        model_dir=MODEL_DIR, dtype="float32", attention_backend=backend, max_len=256))
    result = engine.generate(ids[0].tolist(),
                             SamplingParams(temperature=0, max_new_tokens=10))

    assert result.token_ids == ref_out[: len(result.token_ids)], \
        f"[{backend}] decode unmatched"
