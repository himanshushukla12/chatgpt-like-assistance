"""Step 5: chat history in Redis, trimmed to fit the context window."""
import json
import os

import redis.asyncio as redis

r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

MAX_CONTEXT_TOKENS = 6000  # leave room for the answer inside an 8K window


def _tokens(text: str) -> int:
    return len(text) // 4 + 1  # rough estimate; use the model's tokenizer for exact counts


async def load(chat_id: str) -> list[dict]:
    return [json.loads(m) for m in await r.lrange(f"chat:{chat_id}", 0, -1)]


async def save(chat_id: str, role: str, content: str) -> None:
    await r.rpush(f"chat:{chat_id}", json.dumps({"role": role, "content": content}))
    await r.expire(f"chat:{chat_id}", 60 * 60 * 24)


def trim(history: list[dict], budget: int = MAX_CONTEXT_TOKENS) -> list[dict]:
    """Keep the newest messages that fit the token budget."""
    kept, used = [], 0
    for msg in reversed(history):
        cost = _tokens(msg["content"])
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost
    return list(reversed(kept))
