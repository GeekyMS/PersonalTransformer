# P3 — Training and generation cost

---

## P3.1 Optimizer roofline *(modeling, 2×; AI exact)*

| | Symbolic | Numeric |
|---|---|---|
| bytes moved per step | | |
| FLOPs per step | | |
| **arithmetic intensity** | | |
| side of the ridge | — | |
| **predicted % of step time** | | |

Measured % of step time: ______  Ratio: ______

If far off, why (NumPy temporaries? per-key Python loop?):

>

---

## P3.2 The time budget *(modeling, 2×)*

*Locked on, before the first run:*

| | Value | Source |
|---|---|---|
| FLOPs per step | | P2.1 |
| achieved FLOP/s | | P0.2 |
| **predicted s/step** | | |
| **predicted wall time, 5000 steps** | | |

Measured over 10 steps: ______ s/step

**Implementation efficiency = measured / predicted:** ______

*(Write this number somewhere you'll find it in Phase 5. It's the baseline for the rest of the
project.)*

**Ranked attribution, written before profiling:**

| Rank | Suspected cause | % of gap I think it explains |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

Checked against a profile in Phase 4 — what actually dominated:

>

---

## P3.3 Generation cost *(counting)*

300 characters, `T=256`.

| | Symbolic | Numeric |
|---|---|---|
| FLOPs, recompute-everything (Phase 3.4) | | |
| FLOPs, with a KV cache | | |
| **ratio** | | |
| as a multiple of one training step (8192 tokens) | | |

**Per generated token, with a KV cache:**

| | Value |
|---|---|
| bytes read (weights) | |
| bytes read (cache) | |
| FLOPs | |
| **arithmetic intensity** | |
| side of the ridge | |
| same quantity for a training step at B=32 | |

**Why training and single-stream inference are different problems on the same chip — one
paragraph, my words:**

>
