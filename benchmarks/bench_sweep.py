"""
benchmarks/bench_sweep.py - prefill/decode scaling benchmark

Usage:
  python benchmarks/bench_sweep.py --model ./qwen2.5-0.5b
  python benchmarks/bench_sweep.py --model ./qwen2.5-0.5b \
      --backends naive sdpa --lengths 256 1024 4096
"""
import argparse
import gc
import random

import torch

from minivllm import EngineConfig, LLMEngine, SamplingParams


def make_prompt_ids(engine: LLMEngine, n_tokens: int) -> list[int]:
    vocab = engine.model_cfg.vocab_size
    return [random.randint(0, vocab - 1) for _ in range(n_tokens)]


def measure_once(engine: LLMEngine, prompt_ids: list[int], max_new: int) -> dict:
    """run generate() once and record timing + peak GPU memory"""
    torch.cuda.reset_peak_memory_stats()
    params = SamplingParams(temperature=0, max_new_tokens=max_new)
    res = engine.generate(prompt_ids, params)
    peak = torch.cuda.max_memory_allocated()

    return {
        "prompt_len": res.prompt_len,
        "n_out": len(res.token_ids),
        "prefill_time": res.prefill_time,
        "decode_time": res.decode_time,
        "peak_bytes": peak,
    }


def measure_best(engine: LLMEngine, prompt_ids: list[int], max_new: int, repeats: int = 3) -> dict:
    """warm up once, then keep the fastest of 3 runs"""
    measure_once(engine, prompt_ids, max_new)

    runs = [measure_once(engine, prompt_ids, max_new) for _ in range(repeats)]
    best = dict(runs[0])
    best["prefill_time"] = min(r["prefill_time"] for r in runs)
    best["decode_time"] = min(r["decode_time"] for r in runs)
    best["peak_bytes"] = max(r["peak_bytes"] for r in runs)
    return best


def free_cuda() -> None:
    """release cached blocks so the next engine starts from a clean slate"""
    gc.collect()
    torch.cuda.empty_cache()


def sweep(model_dir: str, backends: list[str], lengths: list[int], max_new: int, repeats: int) -> list[dict]:
    """measure every (backend, prompt length) pair"""
    rows = []
    max_len = max(lengths) + max_new + 8

    for backend in backends:
        engine = LLMEngine(EngineConfig(model_dir=model_dir, attention_backend=backend, max_len=max_len))
        try:
            for n in lengths:
                ids = make_prompt_ids(engine, n)
                try:
                    row = measure_best(engine, ids, max_new, repeats)
                    row["oom"] = False
                except torch.cuda.OutOfMemoryError:
                    free_cuda()
                    row = {"prompt_len": n, "oom": True}
                row["backend"] = backend
                rows.append(row)
        finally:
            del engine
            free_cuda()
    return rows


def print_table(rows: list[dict]) -> None:
    header = (f"{'backend':>8} {'T':>6} {'n_out':>6} "
              f"{'prefill(ms)':>12} {'decode(tok/s)':>14} {'peak(MB)':>10}")
    print()
    print(header)
    print("-" * len(header))

    for r in rows:
        if r["oom"]:
            print(f"{r['backend']:>8} {r['prompt_len']:>6} {'OOM':>6} "
                  f"{'OOM':>12} {'OOM':>14} {'OOM':>10}")
            continue
        tps = r["n_out"] / max(r["decode_time"], 1e-9)
        print(f"{r['backend']:>8} {r['prompt_len']:>6} {r['n_out']:>6} "
              f"{r['prefill_time'] * 1000:>12.2f} {tps:>14.1f} "
              f"{r['peak_bytes'] / 1024 ** 2:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backends", nargs="+", default=["naive", "sdpa"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_sweep needs CUDA")

    rows = sweep(args.model, args.backends, args.lengths, args.max_new_tokens, args.repeats)
    print_table(rows)


if __name__ == "__main__":
    main()
