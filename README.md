# Transformer From Scratch

A decoder-only transformer, implemented three times — **NumPy → C++ → CUDA** — training a
character-level language model on Tiny Shakespeare. The goal isn't the trained model; it's
understanding attention deeply enough that FlashAttention reads as an obvious consequence of the
hardware constraints rather than a clever trick.

Each implementation is validated against the previous one (same seed → same loss curve), and the
NumPy version is itself validated op-by-op and gradient-by-gradient against a PyTorch reference
model.

## Model

Fixed across every implementation so loss curves stay comparable:

| Symbol | Meaning | Value |
|---|---|---|
| `B` | batch size | 32 |
| `T` | context length | 256 |
| `V` | vocab size (character-level) | ~65 |
| `d` | model width | 256 |
| `H` | attention heads | 4 |
| `d_head` | `d / H` | 64 |
| `L` | layers | 6 |

~4.8M parameters. Pre-LN, GELU (tanh approximation), 4× MLP expansion, tied input/output
embeddings, canonical attention layout `(B, H, T, d_head)`.

Out of scope by design: BPE tokenizer, distributed training, mixed precision, dropout, beam
search, encoder-decoder, KV cache (deferred to the CUDA phase).

## Layout

```
oracle/        PyTorch reference model (Phase 0.5)
common/        data loading, gradient checker, FLOP/byte counter — shared across phases
np_impl/       NumPy forward + hand-derived backward + training loop (Phases 1-3)
jax/           JAX/XLA port (Phase 4.5) — not started
cpp/           C++ port: Tensor/Arena/tape autograd, ops, op-by-op test harness (Phase 5)
cuda/          CUDA kernels (Phase 6) — not started
bench/         instrumentation, loss curves, training run artifacts
docs/          roadmap, phase writeups, paper-exercise worksheets
```

## Status

| Phase | What it is | Status |
|---|---|---|
| 0 | Data pipeline, gradient checker, PyTorch oracle, FLOP/byte counter | Done |
| 1 | NumPy forward pass | Done — matches PyTorch oracle to `allclose` 1e-5 |
| 2 | NumPy backward pass, hand-derived | Done — every gradient checked against oracle, 1e-5 |
| 3 | Training loop (AdamW, LR schedule, generation) | Done — see run below |
| 4 | Roofline analysis (arithmetic intensity, ridge point, HBM traffic) | **Scheduled after Phase 6** — no code deliverable on its own; done once CUDA kernels exist to measure, so the analysis explains *why* FlashAttention achieves what it does against real hardware numbers instead of hypothetical ones |
| 4.5 | JAX/XLA port — read the compiled IR, see where it fails to invent FlashAttention | Not started |
| 5 | C++ port with a hand-rolled Tensor/Arena/tape autograd | **In progress** |
| 6 | CUDA kernels (naive → tiled → FlashAttention-style) | Not started |

### Phase 5 detail (current focus)

- `Tensor`, arena allocator, and tape-based autograd are implemented (`cpp/include/`, `cpp/src/`).
- Ops ported and verified against the NumPy reference via the dump/compare harness
  (`cpp/test/README.md`): `matmul`, `embed`, `layer_norm`, `add_bias`, `gelu`, `mlp`,
  attention, and transformer blocks, including backward checks.
- Each op follows the same protocol: implement in C++, dump inputs/outputs to fixed-seed binary
  files, diff against `np_impl` output with `cpp/test/compare.py`. Only move to the next op once
  it passes.
- Full model assembly and the AdamW training loop are implemented. CPU matmul uses BLAS by
  default for forward and backward; the handwritten OpenMP backend remains available.
- Full training comparisons and profiling are ongoing; Phase 5's acceptance gates remain open.

## CPU baseline (NumPy, `bench/run1`)

Initial NumPy run, logged through step 4,900:

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.196 | 4.174 |
| 3,000 (best val) | 1.159 | 1.508 |
| 4,900 (final) | 0.969 | 1.602 |

Untrained loss matches the `ln(V) ≈ 4.17` sanity check. Val loss bottoms out around step 3,000 and
drifts upward after — the run overfits past that point on this dataset size. Loss curve plot:
[`bench/run1/loss_curve.png`](bench/run1/loss_curve.png), full log:
[`bench/run1/loss_log.csv`](bench/run1/loss_log.csv).

**Wall-clock training time:** not recorded for this initial run.

## CPU training performance

Slurm results recorded on October 10, 2026, with four allocated CPUs on AMD EPYC 7763 nodes
and the model constants above:

- Replacing the C++ handwritten OpenMP matmul with OpenBLAS for forward and backward reduced
  the first training update from **175.585 s to 9.435 s: 18.6× faster**. Both runs printed the
  same initial losses (`train 4.2164`, `val 4.1825`).
- Over **5,000 training updates**, the C++ BLAS run was **1.67× faster than NumPy**,
  a **40.2% reduction** in the timed training loop, saving approximately **7h 14m**.

| Implementation | Total timed training loop | Average per update |
|---|---:|---:|
| C++ float32 + OpenBLAS | 38,770.6 s (10h 46m) | 7.7541 s |
| NumPy float64 + OpenBLAS | 64,807.8 s (18h 00m) | 12.9616 s |

Totals include validation, periodic generation, and checkpoint writes; initialization and the
final fixed-seed sample are outside the timer. These compare the current pipelines, with
different precision and generation workloads on separate shared-node allocations. Timing
definitions, BLAS versions, and local correctness checks are in
[`bench/cpu_blas.md`](bench/cpu_blas.md).

## What's next

1. Complete Phase 5 validation and profiling of the full C++ training loop.
2. Validate NumPy/C++ loss curves and fixed-seed samples against the Phase 5 acceptance gate.
3. Phase 4.5 (JAX/XLA) — port forward pass, inspect compiled IR.
4. Phase 6 — CUDA: naive attention kernel, then tiled/online-softmax, working toward a
   FlashAttention-style fused kernel, with a written prediction of what an optimal kernel should
   do *before* reading the paper.
5. Phase 4 (roofline) — once the CUDA kernels exist, use the real measured numbers to explain why
   FlashAttention achieves what it does: arithmetic intensity, the ridge point, HBM round-trips.

