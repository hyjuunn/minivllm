"""
tests/test_batch.py - a batched step must equal the same steps run alone

run:
  export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
  pytest tests/test_batch.py -v
"""
import gc
import os

import pytest
import torch

from minivllm.engine.forward_batch import ForwardBatch
from minivllm.kvcache.simple import SimpleKVCache

MODEL_DIR = os.environ.get("MINIVLLM_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    MODEL_DIR is None, reason="MINIVLLM_TEST_MODEL=<model path> setup needed")

# deliberately uneven, so rows get different amounts of padding
PROMPTS = ["The quick brown fox", "Hello", "In a distant galaxy there once lived a"]
MAX_LEN = 64

_engines = {}


@pytest.fixture(scope="module", autouse=True)
def _release_engines():
    """~2GB of fp32 weights per backend. hand it back
    rather than holding it for the rest of the session like test_generate does"""
    yield
    _engines.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _engine(backend):
    from minivllm import EngineConfig, LLMEngine
    if backend not in _engines:
        _engines[backend] = LLMEngine(EngineConfig(
            model_dir=MODEL_DIR, dtype="float32",
            attention_backend=backend, max_len=MAX_LEN))
    return _engines[backend]


def _cache(engine, batch):
    c = engine.model_cfg
    return SimpleKVCache(c.n_layers, batch, c.n_kv_heads, MAX_LEN,
                         c.head_dim, engine.dtype, engine.device)


def _prefill(engine, cache, ids, slot):
    """prefill one prompt into its slot, return the token it wants next"""
    h = engine.model(torch.tensor([ids], device=engine.device),
                     ForwardBatch.for_prefill(len(ids), slot=slot, device=engine.device),
                     cache)
    return int(engine.model.compute_logits(h[:, -1])[0].argmax())


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_batched_decode_matches_single(backend):
    engine = _engine(backend)
    dev = engine.device
    ids = [engine.tokenizer.encode(p) for p in PROMPTS]
    lens = [len(i) for i in ids]

    with torch.inference_mode():
        # three sequences sharing one cache, then a single B=3 decode step
        shared = _cache(engine, len(ids))
        nxt = [_prefill(engine, shared, i, slot=s) for s, i in enumerate(ids)]
        batched = engine.model(
            torch.tensor(nxt, device=dev).unsqueeze(1),
            ForwardBatch.for_decode(lens, list(range(len(ids))), dev),
            shared)

        # the same three steps, each on its own
        for s, i in enumerate(ids):
            own = _cache(engine, 1)
            _prefill(engine, own, i, slot=0)
            alone = engine.model(torch.tensor([[nxt[s]]], device=dev),
                                 ForwardBatch.for_decode([lens[s]], [0], dev), own)

            diff = (batched[s] - alone[0]).abs().max().item()
            assert diff < 1e-3, (
                f"[{backend}] row {s} (len {lens[s]}, padded to {max(lens) + 1}): "
                f"hidden diff {diff:.2e}")


@pytest.mark.parametrize("backend", ["naive", "sdpa"])
def test_slot_write_does_not_touch_neighbours(backend):
    """the write is addressed by slot, so an untouched row stays zero"""
    engine = _engine(backend)
    cache = _cache(engine, 2)
    ids = engine.tokenizer.encode(PROMPTS[0])

    with torch.inference_mode():
        _prefill(engine, cache, ids, slot=1)

    assert cache.k[0, 1, :, :len(ids)].abs().sum() > 0, "slot 1 was not written"
    assert torch.equal(cache.k[0, 0], torch.zeros_like(cache.k[0, 0])), "slot 0 was clobbered"
