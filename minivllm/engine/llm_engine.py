"""
engine/llm_engine.py - main engine

Current structure: get one request and run prefill + decode loop (simple)
Later: upgrade to step() based
"""

import time
from dataclasses import dataclass, field

import torch

from minivllm.attention import get_attention_backend
from minivllm.config import EngineConfig, ModelConfig
from minivllm.kvcache.simple import SimpleKVCache
from minivllm.loader.weights import load_model
from minivllm.sampling.sampler import SamplingParams, sample
from minivllm.engine.detokenizer import IncrementalDecoder


@dataclass
class GenerationResult:
    text: str
    token_ids: list
    prompt_len: int
    prefill_time: float
    decode_time: float

    @property
    def decode_tps(self) -> float:
        return len(self.token_ids) / max(self.decode_time, 1e-9)

    def summary(self) -> str:
        return (f"prefill {self.prompt_len} tok / {self.prefill_time:.2f}s | "
                f"decode {len(self.token_ids)} tok / {self.decode_time:.2f}s "
                f"= {self.decode_tps:.1f} tok/s")


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = cfg.resolve_device()
        self.dtype = cfg.resolve_dtype(self.device)
        self.model_cfg = ModelConfig.from_pretrained(cfg.model_dir)

        # tokenizer from transformers
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)

        self.backend = get_attention_backend(cfg.attention_backend)
        print(f"[minivllm] device={self.device} dtype={self.dtype} "
              f"backend={type(self.backend).__name__}")

        self.model = load_model(self.model_cfg, cfg, self.backend, self.device, self.dtype)

        # EOS candidates
        self.eos_ids = set()
        # default eos token
        if self.tokenizer.eos_token_id is not None:
            self.eos_ids.add(self.tokenizer.eos_token_id)
        # chat model specific
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            self.eos_ids.add(im_end)

    def _new_cache(self) -> SimpleKVCache:
        # TODO: cfg.kv_cache == "paged"
        c = self.model_cfg
        cache = SimpleKVCache(c.n_layers, 1, c.n_kv_heads,
                              self.cfg.max_len, c.head_dim, self.dtype, self.device)
        return cache

    @torch.inference_mode()
    def generate(self, prompt_token_ids: list, params: SamplingParams, stream_cb=None) -> GenerationResult:
        """core of engine: prefill 1 + decode loop"""
        cache = self._new_cache()
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        prompt_len = input_ids.shape[1]
        assert prompt_len < self.cfg.max_len, "Prompt longer than max_len"

        # --- PREFILL (compute bound) ---
        t0 = time.time()
        hidden = self.model(input_ids, 0, cache)
        logits = self.model.compute_logits(hidden[:,-1])
        next_tok = sample(logits[0], params)
        # wait for gpu
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_time = time.time() - t0

        # --- DECODE (memory-bound) ---
        out_ids = []
        decoder = IncrementalDecoder(self.tokenizer) if stream_cb else None
        pos = prompt_len
        max_steps = min(params.max_new_tokens, self.cfg.max_len - prompt_len - 1)
        t0 = time.time()
        # generation loop
        for _ in range(max_steps):
            if next_tok in self.eos_ids:
                break
            out_ids.append(next_tok)
            if stream_cb:
                piece = decoder.add(next_tok)
                if piece:
                    stream_cb(piece)

            tok = torch.tensor([[next_tok]], device=self.device)
            hidden = self.model(tok, pos, cache)
            logits = self.model.compute_logits(hidden[:, -1])
            next_tok = sample(logits[0], params)
            pos += 1
        if self.device == "cuda":
            torch.cuda.synchronize()
        decode_time = time.time() - t0

        if stream_cb:
            piece = decoder.finalize()
            if piece: 
                stream_cb(piece)

        return GenerationResult(
            text=self.tokenizer.decode(out_ids),
            token_ids=out_ids, prompt_len=prompt_len,
            prefill_time=prefill_time, decode_time=decode_time)

    def chat(self, user_message: str, params: SamplingParams = None,
             stream_cb=None) -> GenerationResult:
        """method for chat template"""
        params = params or SamplingParams()
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True, tokenize=False)
        
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.generate(ids, params, stream_cb=stream_cb)
