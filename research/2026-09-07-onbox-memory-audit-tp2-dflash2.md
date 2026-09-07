## Audit: `vllm-27b-test-dflash2` (GPUs 0+1, TP2, v0.27.1, DFlash2 n=7)

Tonight's boot (05:27 UTC) completed in ~65 s and its profiling lines are identical to the 09-04 boot. Note: a full `docker logs` read stops at 09-05 22:31 (torn line at the 09-06 22:51 stop); `--tail` shows the real end.

### Per-card numbers (measured unless marked est.)

| Item | Value |
|---|---|
| Card / free at start | 23.56 / 23.03 GiB (0.53 pre-existing) |
| Budget, gmu 0.85 | 20.02 GiB |
| Weights + non-torch | 11.61 GiB (weights 11.28; est. target ~9.5 + draft ~1.8 sharded) |
| Peak activation (8192-token chunk + 8-image encoder profile) | 1.19 GiB |
| CUDA graphs, PIECEWISE, 15 sizes | 0.50 GiB, captured after the KV budget, so it eats headroom |
| KV pool | 7.23 GiB = 919 blocks x 8.05 MiB stride -> **324,240 tokens, 1.24x @262K** |
| Block | 1648 tokens (Mamba page 1.60 MiB, padded 0.73%) |
| Per 262K request | 743 blocks = 5.84 GiB: attention 640 (5.03 GiB, ~1.0 GiB of it padding waste), Mamba 90 (0.71), draft SWA 13 (0.10) |
| Amortized | 23.4 KiB/token vs 16 KiB ideal (16 layers x 2 KV heads/card x 256 x K+V x fp8) |
| nvidia-smi now, idle | 21,358 MiB used, 2,767 MiB free |

Checkpoint (safetensors headers): `model.safetensors` 17.41 GiB = int4-packed 11.42 (MLP 8.28, GDN projections 2.68, attention 0.81) + BF16 6.00 (embed 2.37, lm_head 2.37, vision 0.86, GDN norms/A/dt 0.05). `model-mtp.safetensors` 0.79 GiB BF16 (unused with the external drafter). **The draft folder `qwen3.8-27b-dflash2-w4a16` is BF16, not W4A16: 3.58 GiB, 81 tensors, no quantization_config** (MLP 2.49, attention 0.49, conv/selector 0.61).

### Architecture finding
Qwen3.8-27B is a **hybrid**, not dense: 64 layers = 48 `linear_attention` (Gated DeltaNet, 16 K heads x 128, 48 V heads x 128, conv kernel 4) + 16 `full_attention` every 4th; 24 Q / 4 KV heads, head_dim 256, hidden 5120, `mamba_ssm_dtype: float32`, 27-block vision tower. The Flash-Next levers apply; the `max_pixels` cap is already on this lane (1,003,520). Attention KV dominates per-request bytes (86%), so the Mamba-dtype lever pays less here than on Flash-Next.

### Ranked levers (per card; 352.8 tokens per 8.05 MiB block)

1. **gmu 0.85 -> 0.90 / 0.92**: +1.18 / +1.65 GiB -> **+53K / +74K tokens** (377K / 398K). Evidence: log "Free memory on device (23.03/23.56)... 9.59 GiB to fully utilize"; 2.77 GiB free idle. Keep >=1 GiB free (Flash-Next 0.97 died at 310 MiB); the entrypoint strips expandable_segments for `max_split_size_mb:512`, so fragmentation headroom matters. Prefer gmu over `--kv-cache-memory-bytes`: the log's "6.58 GiB to fit" = 7.23 - 0.5 graphs - 0.15, i.e. the explicit budget appears net of the graph reserve (Flash-Next boot 16 lost pool the same way).
2. **A real W4A16 drafter**: today's BF16 draft costs ~1.79 GiB/card. The backport README names a ~1.2 GB syvai W4A16 draft -> est. -1.2 GiB -> **+50K tokens**. Draft attention/MLP accept `quant_config` (qwen3_dflash.py 195-318); only the small DFlash2 projections are pinned to `quant_config=None`. Medium risk: untested load path, verify sha256, acceptance may shift.
3. **`--language-model-only`** (flag exists; interfaces.py:305 skips tower init when limits are 0): -0.43 GiB sharded tower, -80 MiB encoder cache, minus the ViT share of the 1.19 GiB profile (est. 0.3-0.5) -> **+36-45K tokens**. Cost: no images on :8012.
4. **`--mamba-ssm-cache-dtype bfloat16`**: block 1648 -> 880, stride halves; per request 1304 of 1721 blocks -> ~346K (**+22K, +7%**). Needs a needle A/B; the checkpoint pins float32.
5. **`--max-num-batched-tokens 4096`**: peak est. 1.19 -> 0.6-0.8 GiB, draft blocks 13 -> 8 -> **+18-27K**; slower long prefill.
6. CUDA graph trims: fewer capture sizes ~+11K; `--enforce-eager` +22K but decode drops on a launch-bound hybrid. Not recommended.
7. Spec tokens 7 -> 3: Mamba blocks 9 -> 5 per group, ~+21K, but acceptance capped at 4 (09-04 log mean 5-7.5). No.
8. Prefix caching off: align 2 -> 1 state block, +4K, TTFT cost 7x. No.
9. `max-num-seqs`: no pool effect (blocks allocate on demand). No.
10. `kv-cache-dtype`: already fp8_e4m3 on FlashInfer native decode, sm86 (flashinfer.py:824). The backport README's bf16-KV requirement is contradicted by live acceptance; FLASH_ATTN + fp8 crash-loops (09-04 note). No further step on Ampere.
11. EP: n/a, dense MLP.
12. **TP4**: 1,003,062 tokens (3.83x) measured 08-20 on this box, DFlash2 n=7; attention KV/card halves to 8 KiB/token, weights halve. Single-stream slower (85 vs 98 tok/s, PCIe between NVLink pairs). GPUs 2+3 currently hold `heretic-dflash2` at 22.6 GiB each. This is the only lever that roughly triples the pool.

**Structural cost of the external drafter**: its 5 SWA layers set group_size 5, so the 16 attention layers pad into 4 groups of 4 and ~20% of attention KV bytes are dead (~1.0 GiB per 262K request). Without that bucket the same 7.23 GiB would hold roughly 400K. Matches MTP lane 508K vs DFlash2 332K (08-20/09-04 notes). Not a CLI knob.

### Not verified
The code line pinning fp32 SSM (arithmetic and the Flash-Next block halving both point to it); the ViT share of the peak; W4A16-draft loadability under the backport; `--kv-cache-memory-bytes` netting in 0.27.1; the free-memory floor under load (idle only); exact per-card residency of tower and draft (inferred from sharded parallel linears and the 11.28 GiB total).
