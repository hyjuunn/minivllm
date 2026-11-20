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
from minivllm.server.engine_loop import EngineLoop

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    from fastapi import HTTPException
    import uvicorn
except ImportError as e:
    raise SystemExit("try after pip install fastapi uvicorn") from e

app = FastAPI(title="minivllm")
engine: LLMEngine = None
engine_loop: EngineLoop = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = engine.tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        add_generation_prompt=True, tokenize=False)
    ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
    params = SamplingParams(temperature=req.temperature, top_p=req.top_p, max_new_tokens=req.max_tokens)
    # result = engine.generate(ids, params)
    outbox = engine_loop.submit(ids, params)
    toks = []
    while True:
        kind, value = await outbox.get()
        if kind == "token":
            toks.append(value)
        elif kind == "error":
            raise HTTPException(status_code=400, detail=value)
        else:
            # finish
            break

    text = engine.tokenizer.decode(toks)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.cfg.model_dir,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": value,
        }],
        "usage": {
            "prompt_tokens": len(ids),
            "completion_tokens": len(toks),
            "total_tokens": len(ids) + len(toks),
        },
    }


def main():
    global engine, engine_loop
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()
    engine = LLMEngine(EngineConfig(model_dir=args.model, max_len=args.max_len))
    engine_loop = EngineLoop(engine)
    engine_loop.start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
