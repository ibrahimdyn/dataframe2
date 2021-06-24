#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 20G
#SBATCH --time 12:00:00

#SBATCH --output=/home/idayan/slurm-%A_%a.out





module load anaconda
source /home/idayan/anaconda3/bin/activate

cat /etc/hosts
jupyter lab --ip=127.0.0.1 --port=2323
