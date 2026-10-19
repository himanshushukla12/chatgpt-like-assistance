# mini-chatgpt

The code from Part 2 of the @the_aipill "Design ChatGPT" series: a small ChatGPT-style assistant in five steps.

| Step | File | What it does |
|---|---|---|
| 1. Serve | `serve.sh` | Two vLLM servers (OpenAI-compatible): Llama 3.1 8B on :8001, Llama 3.3 70B on :8002 (4 GPUs). PagedAttention is built into vLLM; `--enable-prefix-caching` reuses the KV cache of the shared system prompt. |
| 2. Route | `app/router.py` | Easy prompts → 8B, hard prompts → 70B. A cheap heuristic here; swap in a small trained classifier for production. |
| 3. Stream | `app/main.py` | FastAPI `/chat` endpoint streams tokens back as Server-Sent Events, so the first word shows up fast. |
| 4. Guard | `app/guardrails.py` | Input check before the model, and a streaming output check every few chunks. Regex placeholder; replace with a real safety classifier. |
| 5. Memory | `app/memory.py` | Chat history in Redis (24 h TTL), trimmed to a token budget so it fits the context window. |

## Run

```bash
# GPU box
pip install vllm
./serve.sh

# API (any machine that can reach the GPU box + a Redis)
pip install -r requirements.txt
export REDIS_URL=redis://localhost:6379/0
uvicorn app.main:app --port 8000

curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"chat_id":"demo","message":"What crashes your GPUs first?"}'
```

The stream sends `{"model": ...}` first, then `{"token": ...}` events, then `{"done": true}`.

## Notes and limits

* The 70B model needs about 4×80 GB GPUs in bf16; change `--tensor-parallel-size` or use a quantized checkpoint to fit your hardware. Llama models need an accepted licence on Hugging Face (`huggingface-cli login`).
* Token counts in `memory.py` are a rough `len/4` estimate; use the model's tokenizer for exact budgets.
* Not included (covered in Part 1 as design topics): prefill/decode disaggregation, speculative decoding, autoscaling, auth and rate limits.
* Tested end to end against a mock OpenAI-compatible server with fakeredis (routing, streaming, input guard, history save and trim). Not tested on real GPUs in this environment.
