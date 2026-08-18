"""
tests/test_rope.py - rope table indexing

run:
  pytest tests/test_rope.py -v
"""
import torch

from minivllm.models.layers import RotaryEmbedding, _rotate

HEAD_DIM = 64
MAX_LEN = 1024
THETA = 10000.0


def _qk(B, T, n_heads=4, n_kv_heads=2):
    torch.manual_seed(0)
    return (torch.randn(B, n_heads, T, HEAD_DIM),
            torch.randn(B, n_kv_heads, T, HEAD_DIM))


def test_batch_rows_are_independent():
    """decode step: three sequences at unrelated positions, one gather"""
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN, THETA)
    positions = [37, 512, 8]
    q, k = _qk(B=3, T=1)

    bq, bk = rope(q, k, torch.tensor(positions).unsqueeze(1))  # [3, 1]

    # each row must equal what it would have gotten on its own
    for i, p in enumerate(positions):
        sq, sk = rope(q[i:i + 1], k[i:i + 1], torch.tensor([[p]]))
        assert torch.equal(bq[i:i + 1], sq), f"row {i} (pos {p}): q differs"
        assert torch.equal(bk[i:i + 1], sk), f"row {i} (pos {p}): k differs"


def test_contiguous_gather_matches_slice():
    """prefill: gathering 0..T-1 must reproduce the old cos[:T] slice"""
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN, THETA)
    T = 12
    q, k = _qk(B=1, T=T)

    got_q, got_k = rope(q, k, torch.arange(T).unsqueeze(0))

    cos = rope.cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
    sin = rope.sin[:T].unsqueeze(0).unsqueeze(0)
    assert torch.equal(got_q, _rotate(q, cos, sin))
    assert torch.equal(got_k, _rotate(k, cos, sin))


def test_position_zero_is_identity():
    """t[0] = 0 -> cos 1, sin 0. catches an off-by-one in the table"""
    rope = RotaryEmbedding(HEAD_DIM, MAX_LEN, THETA)
    q, k = _qk(B=1, T=1)

    got_q, got_k = rope(q, k, torch.tensor([[0]]))

    assert torch.equal(got_q, q)
    assert torch.equal(got_k, k)
