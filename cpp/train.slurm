#!/bin/bash
#SBATCH --job-name=transformer-cpp-train
#SBATCH --partition=cpu
#SBATCH --constraint=amd7763
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load cmake/3.28.3-heynjv2
module load gcc/9.4.0

cd "$SLURM_SUBMIT_DIR"

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --target train -j "$SLURM_CPUS_PER_TASK"

# matmul opens a new OpenMP parallel region on every call (~150+ times per
# training step, many of them small per-head matmuls), so fork/join and
# thread-wake latency matter here. OMP_WAIT_POLICY=active keeps worker
# threads spin-waiting between regions instead of sleeping/waking via the OS
# scheduler, and OMP_PROC_BIND pins each thread to a core so it isn't
# migrated between regions -- both reduce that per-call overhead. Pointing
# OMP_NUM_THREADS at the actual allocation keeps it in sync if -c changes.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close

./cpp/build/train
