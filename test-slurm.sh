#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 20G
#SBATCH --time 40:00:00
#SBATCH --array=1-121%3
###  #### SBATCH --output=/zf
### #SBATCH --output=/home/idayan/testt-callallauto.log
### #!/bin/bash
### #SBATCH --ntasks=1
### #SBATCH --partition sixhour
### #SBATCH --time=6:00:00
### #SBATCH --array=1-300

# Job array size will be number of lines in file divided by 
# number of lines chosen below

START=$SLURM_ARRAY_TASK_ID
NUMLINES=100
STOP=$((SLURM_ARRAY_TASK_ID*NUMLINES))
START="$(($STOP - $(($NUMLINES - 1))))"

echo "START=$START"
echo "STOP=$STOP"

for (( N = $START; N <= $STOP; N++ ))
do
    LINE=$(sed -n "$N"p test44.txt)
    echo $LINE
done
