#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 20G
#SBATCH --time 120:00:00
#SBATCH -w helios-cn002
#SBATCH --output=/home/idayan/jupyter.log
# #SBATCH --array=1-128%32





source /home/idayan/env/bin/activate
#export XDG_RUNTIME_DIR=""
#source /home/idayan/anaconda3/bin/activate
#conda activate test

#cat /etc/hosts
jupyter notebook --ip=127.0.0.1 --port=1339
