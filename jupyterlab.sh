#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 20G
#SBATCH --time 12:00:00
#SBATCH --array=1-128%32
#SBATCH --output=/home/idayan/jupyter.log


# #!/bin/bash
# #SBATCH --job-name=jupyter
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --time=2-00:00:00
# #SBATCH --mem=50GB
# #SBATCH --output=/home/<sunetID>/jupyter.log




module load anaconda
source activate /share/sw/open/anaconda/3

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
