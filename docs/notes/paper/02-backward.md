# P2 — The cost of the backward pass

---

## P2.1 Backward:forward ratio *(modeling, 10%)*

For `C = A @ B`:

| | FLOPs |
|---|---|
| forward `C = A @ B` | |
| `dA = dC @ Bᵀ` | |
| `dB = Aᵀ @ dC` | |
| **backward : forward** | |

Ops that violate the ratio, and why (one line each):

| Op | Ratio | Why it differs |
|---|---|---|
| softmax backward | | |
| embedding scatter-add | | |
| layernorm backward | | |
| fused cross-entropy | | |

- Predicted full step (fwd + bwd + optimizer) as a multiple of one forward: ______
- Absolute, at `B=32, T=256`: ______
- Counter: ______  Ratio: ______
- If off by >10%, the op responsible: ______

---

## P2.2 Two adjacent lines *(counting)*

| | `dS = P * (dP - (dP*P).sum)` | `dQ = dS @ K` |
|---|---|---|
| FLOPs | | |
| compulsory bytes | | |
| **arithmetic intensity** | | |
| vs ridge point | | |

- Predicted ratio between the two intensities, **written before computing**: ______
- Actual ratio: ______
- Which one to fix first to speed up the backward pass, and why:

>

---

## P2.3 The retention bill *(counting)*

Every tensor cached in forward for use in backward, at `B=32, T=1024`. Write this **before** any
backward code — it is the spec for what `forward` returns.

| Tensor | Shape | Bytes | Last read at |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| **total** | — | | — |

Largest single cached tensor: ______  (____ % of the total)

**The recompute trade:**

| Quantity | Value |
|---|---|
| extra FLOPs to recompute it in backward | |
| bytes of traffic saved by not storing it | |
| **extra FLOPs per byte saved** | |
| my achieved ridge point (P0.2) | |
| verdict | |

**One sentence, with both numbers in it, dated:**

>

*(In Phase 4.6 you check this sentence against the FlashAttention paper. Don't edit it before then.)*
