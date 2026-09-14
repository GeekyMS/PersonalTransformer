#!/bin/bash
#SBATCH --job-name=transformer-numpy-train
#SBATCH --partition=cpu
#SBATCH --constraint=amd7763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Same partition/-c/--mem/--time as cpp/train.slurm -- same node *type*, for
# a meaningful wall-clock comparison against the C++ trainer. Deliberately
# NOT pinning BLAS to 1 thread: NumPy's multithreaded-BLAS speed over a
# naive scalar C++ matmul is exactly the comparison this is meant to show
# (see roadmap P5.3), not something to normalize away.

set -euo pipefail

# Cap BLAS threading to what SLURM actually allocated (-c 4). Without this,
# OpenBLAS defaults to the node's full physical core count and oversubscribes
# the 4 cores we were cgroup'd into -- especially bad on a shared (non
# --exclusive) node where every other job's BLAS is doing the same thing.
# This does NOT normalize away the multithreaded-BLAS comparison (still 4
# threads, matching -c 4); it just stops threads fighting each other 1-to-N.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Fill in with whatever `module avail python` / `module spider python` shows
# on your account -- same reasoning as cmake/gcc in cpp/train.slurm. Confirm
# numpy is importable after loading (`python3 -c "import numpy"`) -- if not,
# you likely need a venv or a stack/scipy module alongside the bare python one.
module load python/3.11.7

cd "$SLURM_SUBMIT_DIR"

if [ ! -x .venv/bin/python3 ]; then
    python3 -m venv .venv
    .venv/bin/pip install numpy
fi

PYTHONPATH=. .venv/bin/python3 -m np_impl.train
