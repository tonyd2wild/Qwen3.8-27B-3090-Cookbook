# Qwen3.8-27B on RTX 3090 — The Cookbook

**Start here.** Every way we've served **Qwen3.8-27B** on RTX 3090s, which one to pick for your
goal, and the measured findings behind each choice. Each recipe lives in its own repo (linked
below); this page is the map and the decision matrix.

All numbers here are measured on our box — 4× RTX 3090 (24 GB), NVLink pairs GPU0-1 + GPU2-3,
PCIe between the pairs — serving thinking-off, tool-calling agent traffic. Unofficial community
project.

---

## Pick your recipe (decision matrix)

| Your goal | Use | Recipe |
|---|---|---|
| **Default / fastest single-stream** on 2 GPUs | vLLM + **DFlash2**, TP2, 4-bit | [DFLASH2-AutoRound-W4A16-2x3090](https://github.com/tonyd2wild/Qwen3.8-27B-DFLASH2-AutoRound-W4A16-2x3090) |
| **Structured / repetitive throughput** (counting, JSON, codegen) | **SGLang + DSpark**, TP2 | [SGLang-vs-vLLM-2x3090](https://github.com/tonyd2wild/Qwen3.8-27B-SGLang-vs-vLLM-2x3090) |
| **Max context / many long sessions** | vLLM + DFlash2, **TP4** (4 GPUs) | [DFlash2-4x3090-TP4](https://github.com/tonyd2wild/Qwen3.8-27B-DFlash2-4x3090-TP4) |
| **Highest fidelity** (not 4-bit) | vLLM **FP8**, TP2 | [FP8-2x-3090](https://github.com/tonyd2wild/Qwen3.8-27B-FP8-2x-3090) |
| **Uncensored / no-refusal fleet** | DFlash2 TP2 on the **abliterated** weights | recipe = DFLASH2 repo, swap the model + `--quantization compressed-tensors` |
| Same model on **DGX Spark** instead of 3090 | vLLM NVFP4 | [NVFP4-DGX-Spark](https://github.com/tonyd2wild/Qwen3.8-27B-NVFP4-DGX-Spark) |

If you just want it to work and be fast: **start with the DFLASH2 TP2 repo.** That's our
production default.

---

## The recipes

### 1. [DFLASH2-AutoRound-W4A16-2x3090](https://github.com/tonyd2wild/Qwen3.8-27B-DFLASH2-AutoRound-W4A16-2x3090) — the workhorse
vLLM, TP2, AutoRound W4A16 + the **DFlash2** block-diffusion drafter (n=7), FP8 KV, 234K ctx,
tool calling, separated reasoning, vision, CUDA graphs. **~101 tok/s** single-stream decode.
This is what the fleet runs on. Includes the MTP3 vs MTP4 vs DFlash2 comparison.

### 2. [DFlash2-4x3090-TP4](https://github.com/tonyd2wild/Qwen3.8-27B-DFlash2-4x3090-TP4) — the context/concurrency tier
Same model + drafter across **all four** 3090s (TP=4). Full **262K context**, ~**1M-token KV
pool**, **3.83× concurrency** at 262K. The catch: the 4-way all-reduce crosses PCIe between the
NVLink pairs, so single-stream decode is **slower** than TP2 (see finding #3). Use it when you
need context depth and many parallel long sessions, not for raw single-stream speed.

### 3. [SGLang-vs-vLLM-2x3090](https://github.com/tonyd2wild/Qwen3.8-27B-SGLang-vs-vLLM-2x3090) — the engine head-to-head
Same model, same 2 GPUs, both engines minutes apart. **SGLang + DSpark** (a 1.36B draft model)
vs **vLLM + DFlash2/MTP**. Real numbers, plus the KV-pool story that decided it for our fleet.

### 4. [FP8-2x-3090](https://github.com/tonyd2wild/Qwen3.8-27B-FP8-2x-3090) — the fidelity option
vLLM FP8 weights, TP2, tool calling, vision, MTP. Bigger and slower than the 4-bit lanes but
closer to full precision. Use when quality matters more than speed/context.

---

## The findings that actually decide it

**1. DFlash2 vs DSpark splits by workload.** On the same 8-prompt bench (thinking-off, decode
tok/s): SGLang **+ DSpark wins structured/repetitive** output by ~34% (counting, JSON,
repetition). vLLM **+ DFlash2 wins real agent traffic** by +5 to 20% (prose, email, free-form
code) and ties on quicksort. Neither dominates; pick by your traffic. Full table in repo #3.

**2. DFlash2 costs KV pool — the "521K" is a different config.** At **gmu 0.80, TP2**, the KV
pool is **~235K tokens** (AutoRound) / **~263K** (abliterated AWQ, lighter weights). That is
about **1× the 234K context** — one full-length conversation. The oft-quoted **521K** pool was
the older vLLM setup with the **built-in MTP head at gmu 0.90 and no heavy drafter**. The
DFlash2 drafter is a whole separate model in VRAM, so it roughly halves the pool. Don't promise
"2× full context" on a DFlash2 lane.

**3. TP2 beats TP4 for single-stream; TP4 wins KV pool.** TP4's 4-way all-reduce has to cross
PCIe between the two NVLink pairs → single-stream decode drops (~85 vs ~98 tok/s) and saturation
throughput too (146 vs 180). But TP4's KV pool is **~1M tokens (3.83× @262K)** vs TP2's ~259K.
So **TP4 is a context/concurrency tier, not a speed tier.** Two independent TP2 lanes behind a
load-balancer usually beat one TP4 lane for a fleet.

**4. Uncensored costs ~10 eval points — almost all of it by design.** Swapping to the
**abliterated** weights drops the 69-scenario eval from **97.1 → 87.0**, but the drop is
concentrated in **Safety & Boundaries (12→6)** and **Refusal Calibration (10→6)** — categories
that reward *refusing*, which an uncensored model fails on purpose. Real agent skills (tool
selection, structured output, instruction following, multi-step, parameter precision) held
**100% identical**. Cost is real but small: **~11% slower decode** (90 vs 101 tok/s — the
DFlash2 drafter was trained on the base weights, so it accepts fewer draft tokens on the
abliterated ones). Launch is identical to recipe #1 with the abliterated model +
`--quantization compressed-tensors`.

**5. Hardware trap: always `--disable-custom-all-reduce` on these 3090s.** GPU0-1 (NV2) and
GPU2-3 (NV4) are NVLink pairs with PCIe between them. vLLM's custom all-reduce **crashes** this
topology even with NVLink present. Every recipe here ships `--disable-custom-all-reduce`; keep
it. Multi-GPU split also wants `--ipc=host`.

---

## Hardware baseline

- 4× RTX 3090 (24 GB each). NVLink bridges GPU0↔1 and GPU2↔3; the two pairs talk over PCIe.
- A TP2 lane = one NVLink pair (fast intra-pair all-reduce). A TP4 lane spans both pairs and
  eats the PCIe hop.
- Two TP2 lanes + a round-robin relay = our production shape: each new conversation lands on the
  lighter lane, sticks there for prefix-cache warmth, and a dead lane fails over automatically.

## The scoreboard

Every recipe here (plus the Spark fleet) is scored on the same 69-scenario, thinking-off eval:
**[2wild-model-eval.pages.dev](https://2wild-model-eval.pages.dev)** — quality, responsiveness,
deployability, token efficiency, decode tok/s, per model × cluster.

---

*Recipes are independent and stay live; this cookbook is just the front door. Corrections and
your own numbers welcome as issues on the individual repos.*

## 2026-09-07: KV pool ladder on the TP2 + DFlash2 lane (+21.6%, one change per boot)

The live lane (2x RTX 3090 NVLink pair, vLLM v0.27.1, club-3090 DFlash2 backport, 7 draft tokens, fp8 e4m3 KV, 262,144 max-model-len, 6 seats, 1 MP image cap, prefix caching on) started the day at gmu 0.85 with a **324,240-token pool (1.24x at 262K)**. Two research passes (`research/`) found that Qwen3.8-27B is a hybrid (48 Gated DeltaNet + 16 full-attention layers, DeltaNet state pinned to fp32 by the checkpoint), that the profiled activation peak was 1.81 GiB at an 8192-token chunk, and that the lane had 2.8 GiB idle. Each step below is one change on top of the previous one, relaunched through the lane's own compose project (`q27b-dflash2`, `compose/dual/autoround-int4/dflash2.yml` plus an override), health-checked, and probed with a real completion.

| Step | Change | Pool at 262K | Delta | Worker print (per card) | Checks |
|---|---|---:|---:|---|---|
| baseline | gmu 0.85, chunk 8192 | 324,240 (1.24x) | | KV 7.23 GiB, peak 1.19 (older boot) | |
| A1 | gmu 0.90 | 349,290 (1.33x) | +7.7% | consumed 11.61, peak 1.81, graphs 0.49, KV 7.78, idle 1.8 GiB | completion OK |
| A2 | max-num-batched-tokens 4096 | 369,061 (1.41x) | +5.7% | peak 1.81 to 1.50, KV 8.17 | completion OK |
| A3 | `--mamba-ssm-cache-dtype bfloat16` | **394,126 (1.50x)** | +6.8% | unchanged print, block geometry halves | needle correct at 199,584 prompt tokens (TTFT 196 s, ~1,000 tok/s prefill); count-to-100 279 tok/s, prose 64 |

Parked at A3 (Tony's call); A4 = gmu 0.92 is staged and estimated at another +25K with about 1.4 GiB idle. The A3 command list is now the lane's canonical override. Each relaunch takes about 5 minutes because the club entrypoint re-applies its patches.

What the audit says is left, in order: a W4A16 drafter instead of the BF16 DFlash2 draft (about 1.8 GiB per card today, est. +50K, needs a verified download), `--language-model-only` (+36K to 45K, but no image caps on this lane), and TP4 on all four cards (1,003,062 measured 2026-08-20, the only lever that triples the pool). Not worth it here: `--kv-cache-memory-bytes` (net of the graph reserve at v0.27.1, same as the nightly), fewer draft tokens (acceptance is the whole point of DFlash2), prefix caching off (80 MB per seat, 7x TTFT cost), fp8 KV variants (already at the Ampere floor), PP 2x2 (breaks the drafter's embedding sharing).

Caveats: the prose number has no same-method reading of this lane from before the ladder, and DFlash2 accepts poorly on free prose, so treat 64 tok/s as this lane's prose speed rather than a regression until measured against the old config. The bf16 DeltaNet state passed one 200K needle; it is not a full quality equivalence test. One failed attempt earlier in the morning used the wrong compose folder and took the lane down for 26 minutes; the fix was to launch through the project the live container was born from.
