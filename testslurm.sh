#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 12
#SBATCH --time=6:00:00
#SBATCH --array=1-10

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p File.txt)
echo $LINE

#call-program-name-here $LINE
