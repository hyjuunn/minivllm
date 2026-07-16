"""
sampling/sampler.py - sampling param + logic

practice idea: repetition penalty, min_p, multi sequence batch sampling, no aligning logits in gpu..etc
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512


def sample(logits: torch.Tensor, p: SamplingParams) -> int:
    """logits: [vocab] -> next token id"""
    if p.temperature == 0:
        return int(logits.argmax())
    # score to probability
    probs = F.softmax(logits.float() / p.temperature, dim=-1)
    # top_p filtering
    if p.top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumsum - sorted_probs > p.top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        return int(sorted_idx[torch.multinomial(sorted_probs, 1)])
    return int(torch.multinomial(probs, 1))
