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

./cpp/build/train
