#!/bin/bash
#SBATCH --job-name=transformer-numpy-train
#SBATCH --partition=cpu
#SBATCH --constraint=amd7763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Same partition/-c/--mem/--time as cpp/train.slurm -- same node *type*, for
# a meaningful wall-clock comparison against the C++ trainer. Deliberately
# NOT pinning BLAS to 1 thread: NumPy's multithreaded-BLAS speed over a
# naive scalar C++ matmul is exactly the comparison this is meant to show
# (see roadmap P5.3), not something to normalize away.

set -euo pipefail

# Fill in with whatever `module avail python` / `module spider python` shows
# on your account -- same reasoning as cmake/gcc in cpp/train.slurm. Confirm
# numpy is importable after loading (`python3 -c "import numpy"`) -- if not,
# you likely need a venv or a stack/scipy module alongside the bare python one.
module load python/3.11.7
module load py-numpy/1.26.4-kogk3x3

cd "$SLURM_SUBMIT_DIR"

PYTHONPATH=. python3 -m np_impl.train
