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
