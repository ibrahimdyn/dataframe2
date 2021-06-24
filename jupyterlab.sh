#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 20G
#SBATCH --time 12:00:00
#SBATCH --array=1-128%32
#SBATCH --output=/home/idayan/slurm-%A_%a.out
##SBATCH --output=/home/idayan/jupyter.log


# #!/bin/bash
# #SBATCH --job-name=jupyter
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=2-00:00:00
# #SBATCH --mem=50GB
# #SBATCH --output=/home/<sunetID>/jupyter.log




module load anaconda
source /home/idayan/anaconda3/bin/activate

cat /etc/hosts
jupyter lab --ip=127.0.0.1 --port=2323
