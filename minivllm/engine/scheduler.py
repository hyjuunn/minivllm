"""
engine/scheduler.py
TODO
"""
from dataclasses import dataclass, field
from collections import deque

from minivllm.engine.sequence import Sequence, SeqStatus


@dataclass
class SchedulerOutput:
    is_prefill: bool
    seqs: list[Sequence]


class Scheduler:
    def __init__(self, max_batch_size: int = 8):
        self.free_slots: list[int] = list(range(max_batch_size))
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.max_batch_size = max_batch_size

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput | None:
        """decide sequence list to run this step"""
        if self.waiting and self.free_slots:
            # pop front from waiting
            seq = self.waiting.popleft()
            # assign slot
            seq.slot = self.free_slots.pop()
            # add to running
            self.running.append(seq)
            seq.status = SeqStatus.RUNNING
            return SchedulerOutput(is_prefill=True, seqs=[seq])
        
        elif self.running:
            return SchedulerOutput(is_prefill=False, seqs=list(self.running))
        
        return None 

    def finish(self, seq: Sequence, reason: str):
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        # remove from running
        self.running.remove(seq)
        # recover free slot
        self.free_slots.append(seq.slot)
        seq.slot = None

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)
    
