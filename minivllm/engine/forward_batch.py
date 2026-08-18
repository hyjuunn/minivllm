"""
engine/forward_batch.py - per-step metadata handed to the model

Should grow again later: paged KV adds 'block_table' here and no signature changes
~ vLLM's AttentionMetadata / ForwardContext
"""

from dataclasses import dataclass

import torch


@dataclass
class ForwardBatch:

    positions: torch.Tensor   # [B, T] absolute position of each token -> rope does cos[positions]
    slots: torch.Tensor       # [B] kv cache row per sequence
    seq_lens: torch.Tensor    # [B] context length per sequence after this step -> decode attn mask
    is_prefill: bool          # one step is all-prefill or all-decode, never mixed
    max_seq_len: int          # = max(seq_lens) but kept on cpu to prevent D2H sync
    # [B, 1, 1, L] bool - True -> real token, None for prefill
    padding_mask: torch.Tensor | None

    @classmethod
    def for_prefill(cls, prompt_len: int, slot: int, device) -> "ForwardBatch":
        """prompt of length T goes in at positions 0..T-1"""
        return cls(
            positions=torch.arange(prompt_len, device=device).unsqueeze(0), # [1, T]
            slots=torch.tensor([slot], device=device),
            seq_lens=torch.tensor([prompt_len], device=device),
            is_prefill=True,
            max_seq_len=prompt_len,
            padding_mask=None,
        )

    @classmethod
    def for_decode(cls, positions: list[int], slots: list[int], device) -> "ForwardBatch":
        """one new token per sequence, each at its own position"""
        pos = torch.tensor(positions, device=device).unsqueeze(1) # [B, 1]
        seq_lens = pos.squeeze(1) + 1
        L = max(positions) + 1
        mask = torch.arange(L, device=device) < seq_lens.unsqueeze(1) # [B,L]

        return cls(
            positions=pos,
            slots=torch.tensor(slots, device=device),
            seq_lens=seq_lens,
            is_prefill=False,
            max_seq_len=L,
            padding_mask=mask[:, None, None, :], 
        )
