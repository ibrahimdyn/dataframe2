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


#source /home/idayan/env/bin/activate
source /home/idayan/new_env/bin/activate


DATE=$1

printf "%s\n" /zfs/helios/filer1/idayan/$1/* > test_all_.txt

#python /home/idayan//A12_pipeline/pyscripts/FitsFixer.py "/zfs/helios/filer1/idayan/202010030948/2020-10-03T09:56:01-09:59:11_all/SB284-2020-10-03T09:56:01-09:59:11/imgs/*"

#python /home/idayan/dataframe2/AVERAGE/average-frq-corr.py
#python /home/idayan/dataframe2/AVERAGE/averager-meancorr.py
#python /home/idayan/dataframe2/AVERAGE/average.py
