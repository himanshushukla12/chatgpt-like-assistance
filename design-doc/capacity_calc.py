#!/usr/bin/env python3
"""Back-of-envelope capacity calculator for the ChatGPT-like assistant design doc (README.md in this folder).

Change the assumptions at the top and re-run:  python3 capacity_calc.py
Every number is an order-of-magnitude estimate; the method is the point.
"""

# ----------------------------------------------------------------------------------------- assumptions
DAU = 10e6                # daily active users
REQ_PER_USER = 10         # requests per user per day
PROMPT_TOK = 350          # prompt tokens per request after prefix-cache reuse
OUTPUT_TOK = 150          # generated tokens per request
PEAK_FACTOR = 3           # peak / average traffic
STREAM_TOK_S = 30         # tokens/s each user sees while streaming
SMALL_SHARE = 0.8         # share of requests the router sends to the small model
AVG_CTX = 2048            # average context length held in KV during decode
PREFIX_HIT = 0.5          # share of prompt tokens served from the prefix cache
DECODE_EFF = 0.5          # achieved / ideal decode throughput
MFU = 0.5                 # prefill model-FLOPs utilisation
GPU_PRICE = (2.0, 3.0)    # $/hour per H100, low and high
FLEET_HEADROOM = 1.2      # spare capacity for failures and spikes

# hardware: H100 SXM 80 GB
HBM = 80e9
BW = 3.35e12              # bytes/s
BF16_FLOPS = 989e12       # dense
KV_HEADROOM = 0.9         # usable fraction of HBM

# models: (params, layers, kv_heads, head_dim, gpus_per_replica)
LARGE = dict(name="70B", params=70e9, layers=80, kv_heads=8, head_dim=128, gpus=8, batch=256)
SMALL = dict(name="8B", params=8e9, layers=32, kv_heads=8, head_dim=128, gpus=1, batch=192)

GiB = 2**30


def kv_per_token(m, bytes_per_value=2):
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * bytes_per_value


def decode_tok_s_per_replica(m):
    weights = m["params"] * 2
    step = (weights + m["batch"] * AVG_CTX * kv_per_token(m)) / (BW * m["gpus"])
    return m["batch"] / step * DECODE_EFF


def main():
    req_day = DAU * REQ_PER_USER
    avg_rps = req_day / 86400
    peak_rps = avg_rps * PEAK_FACTOR
    tok_req = PROMPT_TOK + OUTPUT_TOK
    avg_tok_s = req_day * tok_req / 86400
    print(f"requests/day        {req_day:,.0f}")
    print(f"avg / peak req/s    {avg_rps:,.0f} / {peak_rps:,.0f}")
    print(f"tokens/day          {req_day * tok_req:,.0f}")
    print(f"avg / peak tok/s    {avg_tok_s:,.0f} / {avg_tok_s * PEAK_FACTOR:,.0f}")
    conc = peak_rps * OUTPUT_TOK / STREAM_TOK_S
    print(f"concurrent streams  {conc:,.0f} at peak (Little's law)\n")

    total = 0.0
    for m, share in ((SMALL, SMALL_SHARE), (LARGE, 1 - SMALL_SHARE)):
        kv = kv_per_token(m)
        rps = peak_rps * share
        dec_tok_s = rps * OUTPUT_TOK
        replicas_bw = dec_tok_s / decode_tok_s_per_replica(m)
        free = HBM * m["gpus"] * KV_HEADROOM - m["params"] * 2
        kv_need = rps * OUTPUT_TOK / STREAM_TOK_S * AVG_CTX * kv
        replicas_kv = kv_need / free
        decode_gpus = max(replicas_bw, replicas_kv) * m["gpus"]
        prefill_gpus = rps * PROMPT_TOK * (1 - PREFIX_HIT) * 2 * m["params"] / (BF16_FLOPS * MFU)
        total += decode_gpus + prefill_gpus
        print(f"[{m['name']}] KV/token {kv / 1024:.0f} KiB | 4K chat {kv * 4096 / 1e9:.2f} GB | "
              f"decode {decode_tok_s_per_replica(m):,.0f} tok/s per replica")
        print(f"      decode GPUs {decode_gpus:,.0f} (bandwidth {replicas_bw * m['gpus']:,.0f}, KV {replicas_kv * m['gpus']:,.0f}) | "
              f"prefill GPUs {prefill_gpus:,.0f}")

    fleet = total * FLEET_HEADROOM
    print(f"\nGPUs at peak ≈ {total:,.0f}; with headroom ≈ {fleet:,.0f}")
    for p in GPU_PRICE:
        print(f"GPU cost at ${p}/h: ${fleet * p / (avg_tok_s * 3600 / 1e6):.2f} per 1M tokens (average load)")


if __name__ == "__main__":
    main()
