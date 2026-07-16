"""
    examples/chat.py - terminal read-eval-print loop

    usage example: 
    
    python examples/chat.py --model ./qwen2.5-0.5b
    python examples/chat.py --model ./qwen2.5-0.5b --backend naive
"""

import argparse
from minivllm import EngineConfig, LLMEngine, SamplingParams


def main():
    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    # save in args
    args = ap.parse_args()

    engine = LLMEngine(
        EngineConfig(
            model_dir=args.model, attention_backend=args.backend, max_len=args.max_len
        )
    )
    params = SamplingParams(
        temperature=args.temperature, max_new_tokens=args.max_new_tokens
    )

    print("Model Loaded.\n")
    while True:
        try:
            user = input("user > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        if user.lower() in {"exit", "quit", "/exit", "/quit"}:
            print("Bye!")
            break
        # streaming
        result = engine.chat(
            user, params,
            stream_cb=lambda t: print(engine.tokenizer.decode([t]), end="", flush=True)
            # flush prevents collecting letters in buffer
        )

        # print summary - (prefill ~ tok / ~ s | decode ~ tok / ~ s = ~ tok/s)
        print(f"\n({result.summary()})\n")


if __name__ == "__main__":
    main()
