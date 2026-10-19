"""Step 4: guardrails on the input, and on the output while it streams.

Kept deliberately small: a regex blocklist. Replace `_flagged()` with a real safety
classifier (a moderation endpoint or a small guard model) for production.
"""
import re

_BLOCK = re.compile(r"\b(make a bomb|credit card dump|steal passwords?)\b", re.I)
REFUSAL = "Sorry, I can't help with that."


def _flagged(text: str) -> bool:
    return bool(_BLOCK.search(text))


def check_input(prompt: str) -> bool:
    """True if the prompt may go to the model."""
    return not _flagged(prompt)


class StreamGuard:
    """Checks the growing answer every `every` chunks, so a bad stream is cut early."""

    def __init__(self, every: int = 8):
        self.every, self.buf, self.n = every, "", 0

    def ok(self, chunk: str) -> bool:
        self.buf += chunk
        self.n += 1
        return self.n % self.every != 0 or not _flagged(self.buf)

    def final_ok(self) -> bool:
        return not _flagged(self.buf)
