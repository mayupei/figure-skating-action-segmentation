#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:15:00
#SBATCH --job-name=train
#SBATCH --mem-per-cpu=10000M
#SBATCH --output=/dev/null

module load python
source ENV/bin/activate

# Run the code on 5 cores.
parallel -j $SLURM_CPUS_PER_TASK '
  python train.py --split {}
' ::: $(seq 0 4)
