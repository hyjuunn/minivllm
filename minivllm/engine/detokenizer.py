"""
engine/detokenizer.py - incremental detokenization for streaming

A single token can hold a partial utf-8 sequence which will decode
into u+fffd
Need to buffer until the text decodes cleanly
"""

class IncrementalDecoder:
    """one per sequence"""

    def __init__(self, tokenizer, window_size: int = 5):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.token_buffer = []
        self.printed_text = ""

    def add(self, token_id: int) -> str:
        """append one token, return newly completed text"""
        self.token_buffer.append(token_id)

        # decode buff
        text = self.tokenizer.decode(self.token_buffer)

        # check if ufffd is in it
        if text.endswith("\ufffd"):
            return ""
        new_text = text[len(self.printed_text):]
        self.printed_text = text

        # use sliding window 
        if len(self.token_buffer) > self.window_size + 2:
            self.token_buffer = self.token_buffer[-self.window_size:]
            self.printed_text = self.tokenizer.decode(self.token_buffer)

        return new_text

    def finalize(self) -> str:
        """flush the tail at end of generation"""
        text = self.tokenizer.decode(self.token_buffer)
        new_text = text[len(self.printed_text):]

        # clear
        self.token_buffer.clear()
        self.printed_text = ""

        return new_text