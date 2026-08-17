# P1 — Counting the forward pass

All four exercises here are **counting**: the bar is exact.

---

## P1.1 Parameter count

*Locked on:*

| Tensor | Shape | Count (symbolic) | Count (numeric) |
|---|---|---|---|
| `tok_emb` | | | |
| `pos_emb` | | | |
| `ln1.g` / `ln1.b` (per block) | | | |
| `attn.Wq/Wk/Wv/Wo` (per block) | | | |
| `ln2.g` / `ln2.b` (per block) | | | |
| `mlp.W1` / `b1` (per block) | | | |
| `mlp.W2` / `b2` (per block) | | | |
| **per block total** | — | | |
| `lnf.g` / `lnf.b` | | | |
| output projection | | | |
| **model total** | — | | |

Measured (`sum(v.size for v in params.values())`): ______  Ratio: ______

- Embedding fraction of total params: ______
- `d` at which embedding fraction drops below 10% (symbolic, then numeric): ______

---

## P1.2 Transpose traffic vs score traffic

*(§1.5 — do this before writing the multi-head reshape.)*

| | Symbolic | T=256 | T=4096 |
|---|---|---|---|
| bytes moved by Q/K/V/O transposes, per layer | | | |
| × L | | | |
| all `S` bytes | | | |
| which dominates | — | | |

- **Crossover `T`, as a formula in `H, dh`:** ______
- Predicted dominant term at T=256 (written before computing): ______
- Predicted dominant term at T=4096: ______

---

## P1.3 Forward FLOPs

Keep the `T²` and `T` groups separate. Do not collapse.

| Op | FLOPs symbolic | Group (`T²` / `T`) | Numeric @ B=32, T=256 |
|---|---|---|---|
| embedding lookup | | | |
| layernorm ×2 | | | |
| Q/K/V projections | | | |
| `Q @ Kᵀ` | | | |
| scale + mask | | | |
| softmax | | | |
| `P @ V` | | | |
| output proj `Wo` | | | |
| MLP `W1` | | | |
| GELU | | | |
| MLP `W2` | | | |
| **per block** | | | |
| **× L, + head** | | | |

- Sum of `T²` terms: ______
- Sum of `T` terms: ______
- **Crossover `T` where they are equal — symbolic first:** ______  numeric: ______
- At `T=256`, dominated by attention or MLP? ______

Counter says: ______  Ratio: ______

---

## P1.4 Memory at three context lengths

`B = 32`.

| | T=256 | T=1024 | T=4096 |
|---|---|---|---|
| parameter bytes | | | |
| retained activation bytes | | | |
| all `S` matrices | | | |

Solve for `T` (algebraically, then numeric):

- `S` bytes = all parameter bytes at `T` = ______
- `S` bytes = 16 GB at `T` = ______
- `S` bytes = my GPU's memory (____ GB) at `T` = ______

Check: the roadmap quotes 8.6 GB for one layer at `B=32, H=4, T=4096`. Does my formula reproduce
it? ______

Counter says: ______  Ratio: ______
