# CPU BLAS backend measurements

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

## Timing method

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

## Correctness

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
