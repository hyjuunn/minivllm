"""
engine/scheduler.py
TODO
"""
from collections import deque

from minivllm.engine.sequence import Sequence


class Scheduler:
    def __init__(self, max_batch_size: int = 8):
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.max_batch_size = max_batch_size

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        """decide sequence list to run this step"""
        raise NotImplementedError("not implemented")
