# P4 — Roofline

**Ground rule: every cell in the "hand" columns is filled in before the counter runs once.**

---

## P4.1 The per-op table

`B=32, T=1024`. Full forward + backward step. *Hand columns locked on:* ______

| Op | FLOPs (sym) | FLOPs (hand) | Compulsory B | Actual B | AI (compulsory) | AI (actual) | **Bound by (predicted)** | Pred. time | Measured FLOPs | Measured bytes | Measured time | Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| embedding | | | | | | | | | | | | |
| layernorm | | | | | | | | | | | | |
| QKV proj | | | | | | | | | | | | |
| `Q @ Kᵀ` | | | | | | | | | | | | |
| scale + mask | | | | | | | | | | | | |
| softmax | | | | | | | | | | | | |
| `P @ V` | | | | | | | | | | | | |
| out proj | | | | | | | | | | | | |
| MLP `W1` | | | | | | | | | | | | |
| GELU | | | | | | | | | | | | |
| MLP `W2` | | | | | | | | | | | | |
| softmax bwd | | | | | | | | | | | | |
| attn matmuls bwd | | | | | | | | | | | | |
| optimizer | | | | | | | | | | | | |

**Ops where compulsory and actual bytes differ most** — this is the Phase 6 work list, in order:

1.
2.
3.

**Bound-by calls I got wrong, and what I mis-modeled:**

>

---

## P4.2 Ridge points and the roofline

| | My GPU | A100 | (H100 / B200 / MI300X) |
|---|---|---|---|
| peak FLOP/s (fp32) | | | |
| peak GB/s | | | |
| **ridge point** | | | |
| source | | | |

- Ridge point trend across generations, as a ratio: ______
- **What that implies for attention's memory-boundedness over time, one sentence:**

>

Hand-drawn log-log roofline: `docs/notes/paper/roofline-hand.jpg` (photograph it, commit it)

---

## P4.3 HBM round-trips over `T²`

| Pass | Count | Line in `np_impl/model.py` |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

- Forward total passes: ______  Backward total passes: ______
- Bytes at `B=32, H=4, T=1024`: ______
- Bytes a perfect kernel would move (one read of Q/K/V, one write of O): ______
- **Factor a fused kernel is playing for:** ______

---

## P4.4 Closed-book estimation exam

Sitting 1: ______  Score: __/8    Sitting 2 (before Phase 6): ______  Score: __/8

| Q | Topic | Answer | Actual | Within 2×? | What I missed |
|---|---|---|---|---|---|
| 1 | score memory at T=8192, freed vs not | | | | |
| 2 | attention fraction at 256 / 1024 / 8192 | | | | |
| 3 | double `d`, halve `L` | | | | |
| 4 | 3 TB/s, 1000 TFLOP/s: ridge, two timings | | | | |
| 5 | Amdahl on the softmax kernel | | | | |
| 6 | bytes eliminated by full fusion | | | | |
| 7 | batch size at the regime boundary | | | | |
| 8 | tile size from 48 KB SMEM (fp32 / fp16 / double-buffered) | | | | |

**Q8 answer carried forward to Phase 6.4:** `Bq = Bk =` ______

---

## P4.6 FlashAttention prediction

**Written before opening the paper. Locked on:** ______

| # | Quantity | My prediction | Paper says | Ratio |
|---|---|---|---|---|
| 1 | HBM traffic as a fraction of my P4.3 count | | | |
| 2 | forward speedup at T=1024 | | | |
| 2b | forward speedup at T=4096 | | | |
| 3 | AI of the fused kernel / side of ridge | | | |
| 4 | what backward does differently + extra FLOP cost | | | |

Cross-check: the sentence I wrote in **P2.3**, months earlier. Was it right?

>

**Where my model was wrong (>2× miss), one paragraph each:**

>

**Where my model was already right** — reread this in Phase 6 when you're tempted to trust the
paper over your own analysis:

>

---

## P4.5.1 Predicting the compiler *(Phase 4.5.3)*

**Written before dumping HLO. Locked on:** ______

| | Predicted | Actual | Notes |
|---|---|---|---|
| # kernels attention becomes | | | |
| ops fused into kernel 1 | | | |
| ops fused into kernel 2 | | | |
| ops fused into kernel 3 | | | |
| largest surviving buffer (shape, bytes) | | | |
| HBM traffic vs my P4.3 count | | | |

**Fusions XLA made that I didn't predict — what property of the dataflow graph made them legal:**

>

**Fusions I predicted that XLA didn't make — why not:**

>
