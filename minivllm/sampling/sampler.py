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


def sample(logits: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    """logits: [B, vocab] -> next token ids [B] (keep on device)"""
    if p.temperature == 0:
        return logits.argmax(dim=-1) #[B]
    # score to probability
    probs = F.softmax(logits.float() / p.temperature, dim=-1) #[B, vocab]
    # top_p filtering
    if p.top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(descending=True) #[B, vocab]
        cumsum = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumsum - sorted_probs > p.top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, idx).squeeze(-1)
    return torch.multinomial(probs, 1).squeeze(-1)
