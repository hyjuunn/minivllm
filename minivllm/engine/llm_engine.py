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
from minivllm.engine.forward_batch import ForwardBatch
from minivllm.engine.scheduler import Scheduler, SchedulerOutput
from minivllm.engine.sequence import Sequence


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

        # scheduler and cache
        self.scheduler = Scheduler(cfg.max_batch_size)
        self.cache = self._new_cache(batch=cfg.max_batch_size)

    def _new_cache(self, batch: int = 1) -> SimpleKVCache:
        # TODO: cfg.kv_cache == "paged"
        c = self.model_cfg
        cache = SimpleKVCache(c.n_layers, batch, c.n_kv_heads,
                              self.cfg.max_len, c.head_dim, self.dtype, self.device)
        return cache

    def add_request(self, prompt_token_ids: list, params: SamplingParams | None = None) -> Sequence:
        """add a new request to scheduler"""
        if params is None:
            params = SamplingParams()
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids is empty")
        if len(prompt_token_ids) >= self.cfg.max_len:
            raise ValueError(f"prompt_token_ids is too long: {len(prompt_token_ids)} >= {self.cfg.max_len}")
        seq = Sequence(prompt_token_ids, params)
        self.scheduler.add(seq)
        return seq

    def _prepare_inputs(self, out: SchedulerOutput) -> tuple[torch.Tensor, ForwardBatch]:
        # Prepare the input tensors and forward batch for the model
        if out.is_prefill:
            # prefill has one sequence
            seq = out.seqs[0]
            input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
            batch = ForwardBatch.for_prefill(input_ids.shape[1], slot=seq.slot, device=self.device)
        else:
            # decode has multiple
            seqs = out.seqs
            input_ids = torch.tensor([[seq.last_token] for seq in seqs], device=self.device)
            batch = ForwardBatch.for_decode(positions=[seq.total_len - 1 for seq in seqs], 
                                            slots=[seq.slot for seq in seqs], 
                                            device=self.device)
        return input_ids, batch

    def _sample(self, seqs: list[Sequence], logits: torch.Tensor) -> list[int]:
        # for every ith sequence in seqs, cut ith row of logits and sample next with its params
        toks = []
        for i, seq in enumerate(seqs):
            toks.append(sample(logits[i:i+1], seq.params))
        return torch.cat(toks).tolist()

    @torch.inference_mode()
    def step(self) -> list[Sequence]:
        """run one step of engine"""
        # ask scheduler
        out = self.scheduler.schedule()
        # if nothing to run, return empty
        if out is None:
            return []
        # prepare inputs
        input_ids, batch = self._prepare_inputs(out)
        hidden = self.model(input_ids, batch, self.cache)
        logits = self.model.compute_logits(hidden[:, -1])
        # sample tokens
        next_toks = self._sample(out.seqs, logits)

        for seq, tok in zip(out.seqs, next_toks):
            # check for EOS
            if tok in self.eos_ids:
                self.scheduler.finish(seq, reason="stop")
            else:
                seq.append_token(tok)
                # if max_new_tokens is reached
                if len(seq.output_token_ids) >= seq.params.max_new_tokens:
                    self.scheduler.finish(seq, reason="length")
                # if max_len is reached (slot is full)
                elif seq.total_len >= self.cfg.max_len:
                    self.scheduler.finish(seq, reason="length")
        return out.seqs

    def generate(self, prompt_token_ids: list, params: SamplingParams, stream_cb=None) -> GenerationResult:
        # add initial req to scheduler
        seq = self.add_request(prompt_token_ids, params)
        # create incremental decoder only if stream_cb
        decoder = IncrementalDecoder(self.tokenizer) if stream_cb else None
        # count streamed tokens
        sent = 0

        def flush():
            nonlocal sent
            if decoder is None:
                return
            for tok in seq.output_token_ids[sent:]:
                piece = decoder.add(tok)
                if piece:
                    stream_cb(piece)
            sent = len(seq.output_token_ids)

        t0 = time.perf_counter()
        self.step()
        flush()
        prefill_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        while not seq.is_finished:
            self.step()
            flush()
        decode_time = time.perf_counter() - t0

        if decoder is not None:
            piece = decoder.finalize()
            if piece:
                stream_cb(piece)

        # return result
        return GenerationResult(
            text=self.tokenizer.decode(seq.output_token_ids),
            token_ids=seq.output_token_ids,
            prompt_len=len(seq.prompt_token_ids),
            prefill_time=prefill_time,
            decode_time=decode_time,
        )

    def chat(self, user_message: str, params: SamplingParams = None,
             stream_cb=None) -> GenerationResult:
        """method for chat template"""
        params = params or SamplingParams()
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True, tokenize=False)
        
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return self.generate(ids, params, stream_cb=stream_cb)
