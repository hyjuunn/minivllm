"""
tests/test_detokenizer.py - incremental detokenization for streaming

run:
  export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
  pytest tests/test_detokenizer.py -v
"""
import os
import random

import pytest

from minivllm.engine.detokenizer import IncrementalDecoder

MODEL_DIR = os.environ.get("MINIVLLM_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    MODEL_DIR is None, reason="MINIVLLM_TEST_MODEL=<model path> setup needed")

CASES = [
    "hello world",
    "안녕하세요",
    "한국어 토크나이저 입니다",
    "값이 誤謬",
    "안녕하세요 반갑습니다 🎉",
    "뷁뻙쒫",
    "a🎉b🎉c",
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _stream(tok, ids, window_size):
    dec = IncrementalDecoder(tok, window_size=window_size)
    return [dec.add(i) for i in ids], dec.finalize()


def _longest_stall(pieces):
    """longest run of consecutive empty returns"""
    longest = run = 0
    for p in pieces:
        run = 0 if p else run + 1
        longest = max(longest, run)
    return longest


@pytest.mark.parametrize("window_size", [1, 5])
@pytest.mark.parametrize("text", CASES)
def test_streams_without_holding_back(tok, text, window_size):
    ids = tok.encode(text)
    pieces, tail = _stream(tok, ids, window_size)

    assert "".join(pieces) + tail == tok.decode(ids)
    assert tail == ""
    assert _longest_stall(pieces) <= 8


@pytest.mark.parametrize("window_size", [1, 5])
def test_random_token_sequences(tok, window_size):
    """the model may emit ids that cut a character anywhere"""
    rng = random.Random(0)
    vocab = len(tok)
    for _ in range(300):
        ids = [rng.randrange(vocab) for _ in range(rng.randint(1, 40))]
        pieces, tail = _stream(tok, ids, window_size)

        assert "".join(pieces) + tail == tok.decode(ids)
