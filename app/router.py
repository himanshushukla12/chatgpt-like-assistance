"""Step 2: a tiny router. Easy prompts -> 8B, hard prompts -> 70B.

This is a cheap heuristic so the repo runs anywhere. In production you would swap
`is_hard()` for a small trained classifier (e.g. a ~100M-param model) and fall back
to the big model when the small one is not confident.
"""
import re

SMALL = {"name": "meta-llama/Llama-3.1-8B-Instruct", "base_url": "http://localhost:8001/v1"}
LARGE = {"name": "meta-llama/Llama-3.3-70B-Instruct", "base_url": "http://localhost:8002/v1"}

HARD_HINTS = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|debug|refactor|optimi[sz]e|algorithm|complexity|"
    r"equation|integral|theorem|architecture|design a|write (a|the) (code|function|program))\b",
    re.I,
)


def is_hard(prompt: str) -> bool:
    return (len(prompt) > 600 or "```" in prompt
            or bool(HARD_HINTS.search(prompt)))


def route(prompt: str) -> dict:
    return LARGE if is_hard(prompt) else SMALL
