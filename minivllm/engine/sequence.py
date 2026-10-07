"""
engine/sequence.py

per-request state the scheduler and engine pass around
"""
import enum
from dataclasses import dataclass, field
from itertools import count
from minivllm.sampling.sampler import SamplingParams

_seq_counter = count()


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    prompt_token_ids: list
    params: SamplingParams
    output_token_ids: list = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    slot: int | None = None
    finish_reason: str | None = None

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SeqStatus.FINISHED

    def append_token(self, token_id: int):
        self.output_token_ids.append(token_id)