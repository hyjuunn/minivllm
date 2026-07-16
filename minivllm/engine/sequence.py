"""
engine/sequence.py

will become fundamental unit managed by scheduler later
TODO
"""
import enum
from dataclasses import dataclass, field
from itertools import count

_seq_counter = count()


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    prompt_token_ids: list
    output_token_ids: list = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    seq_id: int = field(default_factory=lambda: next(_seq_counter))

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]
