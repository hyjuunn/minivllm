"""
benchmarks/bench_generate.py

Usage:
  python benchmarks/bench_generate.py --model ./qwen2.5-0.5b --backends naive sdpa
"""
import argparse
import time

from minivllm import EngineConfig, LLMEngine, SamplingParams

LONG_PROMPT = ("Explain the difference between prefill and decode in LLM inference, "
               "and why one is compute-bound while the other is memory-bound. ") * 8


def bench(model_dir: str, backend: str, max_new: int) -> None:
    engine = LLMEngine(EngineConfig(model_dir=model_dir, attention_backend=backend))
    params = SamplingParams(temperature=0, max_new_tokens=max_new)

    engine.chat("warmup", SamplingParams(temperature=0, max_new_tokens=8))

    t0 = time.perf_counter()
    result = engine.chat(LONG_PROMPT, params)
    total = time.perf_counter() - t0
    print(f"  [{backend:>6}] {result.summary()} | total {total:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backends", nargs="+", default=["naive", "sdpa"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    print(f"prompt length: about {len(LONG_PROMPT.split())} words, "
          f"generate: {args.max_new_tokens} tokens")
    for b in args.backends:
        bench(args.model, b, args.max_new_tokens)


if __name__ == "__main__":
    main()
