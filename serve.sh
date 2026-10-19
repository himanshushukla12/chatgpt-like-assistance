#!/usr/bin/env bash
# Step 1: serve two models with vLLM (OpenAI-compatible API). PagedAttention is built into vLLM;
# --enable-prefix-caching reuses the KV cache of shared prefixes (e.g. the system prompt).
set -euo pipefail

# small, fast model for easy prompts (1 GPU)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8001 --enable-prefix-caching --gpu-memory-utilization 0.90 &

# big reasoning model for hard prompts (4 GPUs, tensor parallel)
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --port 8002 --tensor-parallel-size 4 --enable-prefix-caching &

wait
