"""
server/api.py - OpenAI compatible API server

run:
  pip install fastapi uvicorn
  python -m minivllm.server.api --model ./qwen2.5-0.5b --port 8000

test:
  curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"hi!"}]}'
"""
import argparse
import time
import uuid

from minivllm import EngineConfig, LLMEngine, SamplingParams

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    raise SystemExit("try after pip install fastapi uvicorn") from e

app = FastAPI(title="minivllm")
engine: LLMEngine = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    prompt = engine.tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True, tokenize=False)
    ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
    params = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_new_tokens=req.max_tokens)
    result = engine.generate(ids, params)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.cfg.model_dir,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.prompt_len,
            "completion_tokens": len(result.token_ids),
            "total_tokens": result.prompt_len + len(result.token_ids),
        },
    }


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    engine = LLMEngine(EngineConfig(model_dir=args.model, max_len=args.max_len))
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
