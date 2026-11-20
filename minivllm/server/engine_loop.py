"""
engine_loop.py - engine loop to run LLMEngine in a separate thread
"""

import asyncio
import queue
import threading
from minivllm.engine.llm_engine import LLMEngine


class EngineLoop:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.inbox: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, prompt_ids, params) -> asyncio.Queue:
        outbox = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self.inbox.put((prompt_ids, params, outbox, loop))
        return outbox

    def _send(self, outbox, loop, msg):
        loop.call_soon_threadsafe(outbox.put_nowait, msg)

    def _admit(self, ticket, live):
        ids, params, outbox, loop = ticket
        try:
            seq = self.engine.add_request(ids, params)
        except ValueError as e:
            self._send(outbox, loop, e)
            return
        live[seq.seq_id] = [seq, outbox, loop, 0]
    
    def _run(self):
        # main engine thread loop
        live = {} # seq_id -> [seq, outbox, loop, sent]

        while True:
            # check inbox
            if self.inbox.empty():
                request = self.inbox.get()
            else:
                request = self.inbox.get_nowait()
            


