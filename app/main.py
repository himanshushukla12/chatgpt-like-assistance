"""Step 3: FastAPI streams tokens back (Server-Sent Events), so the first word shows up fast.

Run:  uvicorn app.main:app --port 8000
Test: curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
           -d '{"chat_id":"demo","message":"What crashes your GPUs first?"}'
"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from . import memory
from .guardrails import REFUSAL, StreamGuard, check_input
from .router import route

SYSTEM = {"role": "system", "content": "You are a helpful, concise assistant."}  # shared prefix -> prefix cache hit

app = FastAPI(title="mini-chatgpt")
_clients: dict[str, AsyncOpenAI] = {}


def client_for(base_url: str) -> AsyncOpenAI:
    if base_url not in _clients:
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    return _clients[base_url]


class ChatIn(BaseModel):
    chat_id: str
    message: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/chat")
async def chat(req: ChatIn):
    async def events():
        if not check_input(req.message):  # input guardrail
            yield sse({"token": REFUSAL})
            yield sse({"done": True})
            return

        model = route(req.message)  # 8B or 70B
        yield sse({"model": model["name"]})

        history = memory.trim(await memory.load(req.chat_id))
        messages = [SYSTEM, *history, {"role": "user", "content": req.message}]

        guard, answer = StreamGuard(), []
        stream = await client_for(model["base_url"]).chat.completions.create(
            model=model["name"], messages=messages, stream=True, max_tokens=512
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if not token:
                continue
            if not guard.ok(token):  # output guardrail, while streaming
                yield sse({"token": "\n" + REFUSAL})
                break
            answer.append(token)
            yield sse({"token": token})

        await memory.save(req.chat_id, "user", req.message)
        await memory.save(req.chat_id, "assistant", "".join(answer))
        yield sse({"done": True})

    return StreamingResponse(events(), media_type="text/event-stream")
