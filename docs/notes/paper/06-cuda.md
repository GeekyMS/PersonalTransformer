# P6 — CUDA

**P6.0: every kernel gets a prediction written before the kernel is.**

---

## The benchmark table

| Kernel | FLOPs | Compulsory bytes | AI | Bound by | Pred. time | Pred. GB/s | Pred. speedup | **Measured time** | **Measured GB/s** | **% peak** | **Actual speedup** | Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6.1 naive | | | | | | | — | | | | — | |
| 6.2 fused softmax | | | | | | | | | | | | |
| 6.3 online softmax | | | | | | | | | | | | |
| 6.4 FA forward | | | | | | | | | | | | |
| 6.5 FA backward | | | | | | | | | | | | |
| 6.6 KV cache | | | | | | | | | | | | |

**Post-mortem for every miss > 2×** — was the model wrong or the kernel bad? Name the mechanism
(occupancy, launch overhead, uncoalesced access, bank conflicts, register spills):

| Kernel | Miss factor | Model wrong / kernel bad | Mechanism |
|---|---|---|---|
| | | | |
| | | | |

---

## P6.1 Naive baseline traffic

- Predicted total HBM traffic: ______
- Same count for NumPy from P4.3: ______  (should agree symbolically)
- Predicted runtime at achieved bandwidth: ______  Measured: ______

---

## P6.3 Online softmax, on paper

**1. Exactness.** Induction over tiles: carrying `(m, l)` and rescaling on max update gives the
same result as the single pass in exact arithmetic.

>
>
>

**2. Error.** Relative fp32 error bound of the online version vs the two-pass version:

>

- Which is more accurate, and why: ______
- **Derived test tolerance for 6.4:** ______ (derived, not tuned until the test passed)

---

## P6.4 Tile size

From exam Q8 (capacity only): `Bq = Bk =` ______

Redone with real constraints:

| Constraint | Value |
|---|---|
| usable SMEM per SM | |
| registers per thread budget | |
| target occupancy | |
| **resulting `Bq = Bk`** | |
| moved from the Q8 answer? | |

**Halving the tile — predicted before measuring.** Two competing effects:

| Effect | Direction | Magnitude |
|---|---|---|
| K/V re-reads from HBM | | |
| occupancy / latency hiding | | |
| **which wins, and why** | | |

Measured, full tile: ______  half tile: ______  Was I right? ______

---

## P6.6 Inference throughput

| | Batch 1 | Batch 32 |
|---|---|---|
| weight bytes read per token | | |
| KV cache bytes read per token | | |
| FLOPs per token | | |
| **predicted tokens/sec** | | |
| measured tokens/sec | | |
| ratio | | |

- Throughput as a function of batch size (sketch): ______
- Per-token latency as a function of batch size: ______
- **Where the curve bends, and why:** ______
- Explanation of the predicted↔measured gap:

>
