#!/bin/bash
#SBATCH --job-name=transformer-cpp-train
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Unity modules are versioned (module load <name>/<version>) -- run
# `module avail cmake gcc` on Unity to find the exact strings available on
# your account, then fill these in. Leaving unversioned `module load cmake`/
# `module load gcc` may fail or silently pick an unexpected default.
module load cmake/<version>
module load gcc/<version>

cd "$SLURM_SUBMIT_DIR"

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target train -j "$SLURM_CPUS_PER_TASK"

./cpp/build/train
