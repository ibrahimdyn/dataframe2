#!/bin/bash 
### ### #SBATCH --partition neutron-star
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 5
#SBATCH --mem 60G
### ## #SBATCH --time 240:00:00
##### ###SBATCH --array=13-17
#### #SBATCH --array=1-650%200
#### #SBATCH --exclude=helios-cn[016-020]
### ###SBATCH --output=/home/idayan/LOGALL/AVERAGE-PROCESS.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))
FILE =$1

python /home/idayan/dataframe2/AVERAGE/timestamps_filter_TEST.py FILE
