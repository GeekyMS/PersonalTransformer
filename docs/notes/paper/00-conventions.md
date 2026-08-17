# P0 — Conventions and hardware

Everything downstream inherits from this file. Fill it in once, date any change, and redo every
affected number if you change one.

---

## P0.1 Costing conventions

*Locked on:*

| Decision | Choice | Why / source |
|---|---|---|
| FMA counted as | | |
| Vendor peak FLOP/s uses which convention | | |
| `exp` costs | | |
| `tanh` costs | | |
| `sqrt` / `rsqrt` cost | | |
| SFU : FMA throughput ratio on my GPU | | |
| Compulsory bytes — definition I'm using | | |
| Actual bytes — definition I'm using | | |
| Do op byte counts include their parameters? | | |
| How parameter bytes scale with `B` | | |

### Per-op formulas

Both byte conventions. Symbolic only — no numbers in this table.

| Op | FLOPs | Compulsory bytes | Actual bytes (my impl) |
|---|---|---|---|
| matmul `(M,K)@(K,N)` | | | |
| row-softmax `(R,C)` | | | |
| layernorm `(N,d)` | | | |
| GELU, `N` elements | | | |
| elementwise add, `N` elements | | | |

**Gate:** the `count()` calls in `common/counter.py` are transcriptions of the rows above.

---

## P0.2 Hardware sheet

| | Device A (CPU) | Device B (GPU) |
|---|---|---|
| Name | | |
| Clock | | |
| Cores / SMs | | |
| FMA width (lanes × ops) | | |
| **Peak fp32 FLOP/s** (show the product) | | |
| Memory clock × bus width × channels | | |
| **Peak GB/s** (show the product) | | |
| L1 / SMEM per core | | |
| L2 | | |
| L3 / none | | |
| **Ridge point (spec)** | | |

### Measured

| | Device A | Device B | Method |
|---|---|---|---|
| Achieved FLOP/s | | | large square matmul, size = |
| Achieved GB/s | | | triad `a[i] = b[i] + s*c[i]`, N = |
| **Achieved ridge point** | | | |
| Achieved / spec, compute | | | |
| Achieved / spec, bandwidth | | | |

**Why achieved differs from spec — one sentence per axis, per device:**

-
-

**Gate:** I can state the achieved ridge point of each device from memory.
