# P5 — C++ port

---

## P5.1 Arena sizing *(counting)*

Peak-usage point in the step (which op, forward or backward): ______

| Buffer | Shape | Bytes | Live from → to |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| **naive sum = arena size** | — | | — |

- Theoretical minimum with aggressive reuse of dead buffers: ______
- **Gap, as a %:** ______ ← this is what a compiler's buffer-assignment pass would recover
- Allocator high-water mark, measured: ______  Ratio: ______

---

## P5.2 Cache blocking *(modeling, 2×)*

Working set of a `Tq × Tk` tile of the `(T,dh) × (dh,T)` matmul:

- Capacity inequality (symbolic): ______
- Largest tile fitting L1 (____ KB): ______
- Largest tile fitting L2 (____ KB): ______
- Tile arithmetic intensity at that size: ______
- Achieved bandwidth at that level of the hierarchy: ______
- **Predicted GFLOP/s:** ______

Measured: ______  Ratio: ______

Cross-check against the tiled matmul numbers in `cuda-kernels` — does the same reasoning reproduce
them?

>

---

## P5.3 Port speedup *(modeling, 2× per category)*

**Predicted before running:**

| Category | Predicted speedup vs NumPy | Actual | Why |
|---|---|---|---|
| matmuls (NumPy calls BLAS — be honest) | | | |
| elementwise ops | | | |
| softmax | | | |
| optimizer step | | | |
| per-op Python overhead | | | |
| **whole step** | | | |

- P3.2 implementation efficiency, before: ______  after the port: ______
- **Fraction of the gap the port actually closed:** ______
- Categories where I correctly predicted no gain: ______
