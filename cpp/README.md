# CPU build

BLAS is the default for matmul forward and both backward products. The
`matmul(A, B, out)` interface and tape stay the same. Forward overwrites the
output; backward accumulates into existing gradients.

```sh
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j 4
```

On macOS 13.3+ this uses Apple Accelerate. On Linux it uses OpenBLAS, including
its development headers (`cblas.h`). On the Slurm server, check the available
OpenBLAS modules:

```sh
module avail openblas
```

If OpenBLAS is not already available, add `module load` with the exact name
from that output to `cpp/train.slurm`, after its GCC module load and before
CMake. The module name and version depend on the cluster. If CMake already
finds OpenBLAS, no extra module load is needed. The script sets the BLAS
thread count to `SLURM_CPUS_PER_TASK`.

For a library installed outside the usual search paths, pass
`-DCMAKE_PREFIX_PATH=/path/to/openblas`. You can also select a different
CBLAS provider with `-DBLA_VENDOR=...` and set `-DCBLAS_INCLUDE_DIR=...`.
CMake checks that the selected library supplies `cblas_sgemm`.

The current handwritten OpenMP matmul remains available:

```sh
cmake -S cpp -B cpp/build-naive -DCMAKE_BUILD_TYPE=Release -DUSE_BLAS=OFF
cmake --build cpp/build-naive -j 4
```

BLAS calls have no surrounding OpenMP parallel region. Row-major views,
transposes, offsets, and padded row strides are handled directly; other
strided views are packed for BLAS and output is scattered back afterward.

## CPU training performance

On Slurm with four allocated CPUs, OpenBLAS reduced the first C++ training
update from **175.585 s to 9.435 s**, an **18.6× speedup** over the
handwritten OpenMP backend. The initial printed losses stayed the same.

The completed 5,000-update C++ BLAS run took **38,770.6 s (10h 46m)**,
averaging **7.7541 s/update**. NumPy took **64,807.8 s (18h 00m)**,
averaging **12.9616 s/update**: C++ was **1.67× faster**, with **40.2% less
time** in the timed training loop.

These runs use different precision and generation workloads on separate
shared-node allocations. See [bench/cpu_blas.md](../bench/cpu_blas.md) for
the configuration, timing definitions, and correctness checks.
