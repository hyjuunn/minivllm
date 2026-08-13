# minivllm

A minimal vLLM-style LLM inference engine built from scratch for study - paged KV cache, a batching scheduler, pluggable attention backends, and an OpenAI-compatible server (Paged KV cache / batching: WIP).

## Quick start

```bash
# Install (CUDA torch + triton pulled in automatically)
pip install -e ".[dev,server]"

# Download the model (~1GB)
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./qwen2.5-0.5b

# Chat
python examples/chat.py --model ./qwen2.5-0.5b

# Run tests
export MINIVLLM_TEST_MODEL=./qwen2.5-0.5b
pytest tests/ -v

# Benchmark attention backends
python benchmarks/bench_generate.py --model ./qwen2.5-0.5b --backends naive sdpa

# OpenAI-compatible server
python -m minivllm.server.api --model ./qwen2.5-0.5b
```
