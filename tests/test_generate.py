"""
tests/test_generate.py - the decode loop itself.

test_correctness.py drives engine.model directly, so generate() - position
bookkeeping, cache reuse, eos handling, streaming - is never exercised.
Numerics vs HF are covered there; this file only checks the loop.

run:
  export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
  pytest tests/test_generate.py -v
"""
import os

import pytest
import torch

MODEL_DIR = os.environ.get("MINIVLLM_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    MODEL_DIR is None, reason="MINIVLLM_TEST_MODEL=<model path> setup needed")

PROMPT = "The quick brown fox"
MAX_NEW = 16

_engines = {}


def _engine(backend):
    from minivllm import EngineConfig, LLMEngine
    if backend not in _engines:
        _engines[backend] = LLMEngine(EngineConfig(
            model_dir=MODEL_DIR, dtype="float32",
            attention_backend=backend, max_len=256))
    return _engines[backend]


def _greedy(engine, params):
    from minivllm import SamplingParams
    return engine.generate(engine.tokenizer.encode(PROMPT), params)


def _reference_loop(engine, prompt_ids, max_new_tokens):
    """prefill + decode written out by hand, the way generate() should behave"""
    cache = engine._new_cache()
    ids = torch.tensor([prompt_ids], device=engine.device)
    out = []
    with torch.inference_mode():
        hidden = engine.model(ids, 0, cache)
        nxt = int(engine.model.compute_logits(hidden[:, -1])[0].argmax())
        pos = len(prompt_ids)
        for _ in range(max_new_tokens):
            if nxt in engine.eos_ids:
                break
            out.append(nxt)
            step = torch.tensor([[nxt]], device=engine.device)
            hidden = engine.model(step, pos, cache)
            nxt = int(engine.model.compute_logits(hidden[:, -1])[0].argmax())
            pos += 1
    return out


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_matches_reference_loop(backend):
    from minivllm import SamplingParams
    engine = _engine(backend)
    prompt_ids = engine.tokenizer.encode(PROMPT)

    got = engine.generate(prompt_ids, SamplingParams(temperature=0, max_new_tokens=MAX_NEW))
    want = _reference_loop(engine, prompt_ids, MAX_NEW)

    assert got.token_ids == want
    assert got.prompt_len == len(prompt_ids)


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_greedy_is_deterministic(backend):
    from minivllm import SamplingParams
    engine = _engine(backend)
    p = SamplingParams(temperature=0, max_new_tokens=MAX_NEW)

    # temperature=0 must take the argmax path and never touch the rng
    assert _greedy(engine, p).token_ids == _greedy(engine, p).token_ids


def test_respects_max_new_tokens():
    from minivllm import SamplingParams
    engine = _engine("sdpa")
    for n in [1, 5]:
        r = _greedy(engine, SamplingParams(temperature=0, max_new_tokens=n))
        assert len(r.token_ids) <= n


def test_stops_at_eos():
    from minivllm import SamplingParams
    engine = _engine("sdpa")
    p = SamplingParams(temperature=0, max_new_tokens=MAX_NEW)
    full = _greedy(engine, p).token_ids

    # declare a token the model is about to emit as eos
    target = full[3]
    cut = full.index(target)

    saved = engine.eos_ids
    engine.eos_ids = saved | {target}
    try:
        stopped = _greedy(engine, p).token_ids
    finally:
        engine.eos_ids = saved

    assert stopped == full[:cut]


def test_streamed_text_matches_result():
    from minivllm import SamplingParams
    engine = _engine("sdpa")
    pieces = []
    r = engine.generate(engine.tokenizer.encode("한국어로 인사해줘 🎉"),
                        SamplingParams(temperature=0, max_new_tokens=MAX_NEW),
                        stream_cb=pieces.append)

    assert "".join(pieces) == r.text
    assert "�" not in "".join(pieces)
