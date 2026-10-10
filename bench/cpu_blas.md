# CPU BLAS backend measurements

## Local measurements (October 8, 2026)

Measured locally on October 8, 2026, on an Apple M3 with AppleClang 21,
Release (`-O3`) builds, float32 tensors, and the current model at
`B=32, T=256, d=256, H=4, L=6`. These are local measurements, not Slurm
server timings or a full training convergence test.

The same sources were built three ways: the retained handwritten OpenMP
backend, Apple Accelerate, and OpenBLAS 0.3.28. Thread limits of four were
requested through `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. OpenBLAS reports four
threads with these settings; its installed build uses OpenMP internally.
Accelerate's actual worker count was not inspected.

### Timing method

A temporary harness called the existing dataset, initialization, forward,
loss, backward, gradient clipping, AdamW, and evaluation functions. Each
backend ran sequentially in a fresh process with the same seed and first
batch. Timing used `std::chrono::steady_clock`, without debugger stops.
Initialization and binary dump writes were outside the update timer.
The optimizer used `lr_at(1, 5000)` so that the weight comparison exercised
a nonzero update; the actual trainer uses zero learning rate at step 0.

One update and two validation batches were timed per backend. These single
samples include initial memory touches and are not steady-state medians.

| Backend | Forward | Backward | Full update | Validation batch, mean of two |
|---|---:|---:|---:|---:|
| Handwritten OpenMP | 10.790 s | 54.494 s | 65.310 s | 11.686 s |
| Apple Accelerate | 0.901 s | 1.230 s | 2.157 s | 0.669 s |
| OpenBLAS | 0.901 s | 1.463 s | 2.387 s | 0.830 s |

Full-update speedups were approximately 30.3x for Accelerate and 27.4x
for OpenBLAS relative to the fresh OpenMP run.

The actual Accelerate trainer was also launched in an isolated directory
and stopped immediately after its first loss line. It printed after
21.01 seconds of wall time, including startup and its 20 validation batches:

```text
0: train 4.2234  val 4.1920  (2.323 s/step, 2.3s elapsed)
```

### Correctness

All 17 op contracts passed against NumPy with each backend: 51 comparisons
total. This includes attention and transformer block backward checks.
The new `matmul_backward` fixture checks 64 combinations of contiguous,
padded, transposed, and general strided views, nonzero offsets, nonzero
initial gradients, a second backward accumulation, untouched padding, and
the absence of additional tape nodes during backward.

For the full-size training batch, all clipped parameter gradients and
updated weights were compared against the handwritten backend using
`atol=1e-6, rtol=1e-4`:

| Backend | Maximum gradient difference | Maximum updated-weight difference |
|---|---:|---:|
| Apple Accelerate | 1.49e-8 | 5.84e-8 |
| OpenBLAS | 3.91e-8 | 9.80e-8 |

Build and backend selection instructions are in [cpp/README.md](../cpp/README.md).
Op fixture generation and comparison commands are in
[cpp/test/README.md](../cpp/test/README.md).

The temporary harnesses, raw measurements, and isolated build directories
are in `/private/tmp/pt-blas-MSVWfF`; they are not required to build or train.

## Slurm training results (recorded October 10, 2026)

These measurements come from the reported Slurm `.out` logs, rather than
the local harness above. The jobs requested four CPUs, one task, 16 GB of
memory, and the `amd7763` constraint (AMD EPYC 7763), without an exclusive
node allocation. Training used `B=32, T=256, d=256, H=4, L=6`, with BLAS
thread limits set to four.

The C++ build used Release optimization (`-O3 -DNDEBUG`) and the
`openblas/0.3.28` module. NumPy was version 2.4.6; its BLAS configuration
reported `scipy-openblas` version `0.3.31.188.0`. The completed NumPy run
was on `cpu026`, with CPU affinity `[107, 108, 109, 110]`.

### C++ before and after BLAS

| Matmul backend | Step 0 update time | Printed train loss | Printed validation loss |
|---|---:|---:|---:|
| Handwritten OpenMP | 175.585 s | 4.2164 | 4.1825 |
| OpenBLAS, forward and backward | 9.435 s | 4.2164 | 4.1825 |

The observed first-update speedup was **18.6×** (`175.585 / 9.435`). These
are single first-update measurements; no complete pre-BLAS Slurm run was
recorded for an equivalent full-run comparison. Matching the printed
losses supports the comparison; the local gradient and weight checks above
provide more detailed correctness evidence.

### Completed C++ and NumPy runs

Both runs completed 5,000 training updates, evaluating 20 validation
batches every 100 updates and generating 300 tokens plus a checkpoint
every 500 updates, starting at step 0.

| Implementation | Precision | Total timed training loop | Average per update | Step 4,900 train / validation loss |
|---|---|---:|---:|---:|
| C++ + OpenBLAS | float32 | 38,770.6 s (10h 46m) | 7.7541 s | 0.9761 / 1.6144 |
| NumPy + OpenBLAS | float64 | 64,807.8 s (18h 00m) | 12.9616 s | 0.9687 / 1.6022 |

The C++ timed loop was **1.67× faster** (`64,807.8 / 38,770.6`), taking
**40.2% less time** and saving **26,037.2 s**, approximately **7h 14m**.
The final reported non-generation intervals were around 5.4–5.5 s/update
for C++ and 13.1–13.3 s/update for NumPy; these intervals still include
validation overhead.

### What the timers include

- Step 0's `s/step` measures the first training update. The timer is read
  before validation, although the loss line prints after validation finishes.
- Later `s/step` values average the interval since the previous timer read
  over 100 updates. They include the previous validation and any generation
  or checkpoint writes during that interval. For example, step 4,600
  includes generation after step 4,500.
- `total training time` includes all 5,000 updates, periodic validation,
  generation, and checkpoint writes, including the final checkpoint.
  Initialization and the final fixed-seed 500-token sample are outside
  this timer. Compilation and Slurm queue time are also excluded.

### Comparison limits

The full-run ratio measures the current implementations on separate
shared-node allocations. It includes precision, BLAS build, and pipeline
differences: C++ uses float32, while NumPy uses float64. NumPy generates
one sequence using its current context length; C++ generates with 32
duplicated sequences padded to length 256. Initialization RNGs also differ,
so setting the same seed does not produce identical initial parameters
across languages. These results alone do not establish the Phase 5
loss-curve and fixed-seed generation acceptance gates.

An earlier NumPy run on `cpu023` printed 59.998 s for step 0 and varied
widely during training. It was excluded from the completed-run comparison.
The retry on `cpu026` printed 12.459 s for step 0 and completed with the
timings above. Diagnostics on the slow allocation showed four threads,
affinity spanning two sockets and three NUMA nodes, no process swap, and
no cgroup CPU-quota throttling in the sampled counters. Its exact slowdown
cause was not established; server profiler samples were not available.
