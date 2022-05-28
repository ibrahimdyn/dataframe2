#!/bin/bash 
#SBATCH -N 1
## #SBATCH --partition neutron-star
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 40G
### ## ##SBATCH --time 05:00:00
## ####SBATCH --output=/home/idayan/noisedist.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))

#source /home/idayan/env/bin/activate
source /home/idayan/new_env/bin/activate

echo "processing first folder:"
echo $1

#cp /zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits /hddstore/idayan/
#python /home/idayan/dataframe2/crtdataframe.py /home/idayan/imglst/*.fits
#python /home/idayan/dataframe2/crtdataframe.py $SLURM_ARRAY_TASK_ID"/zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits"


#python /home/idayan/dataframe2/crtdataframe.py 
#python /home/idayan/dataframe2/AllDatesDataFrame.py
#python /home/idayan/dataframe2/AllDateDF-Automat.py
#CALIBRATION202008122000.py
#python /home/idayan/dataframe2/CALIBRATION202008122000.py
#python /home/idayan/dataframe2/justcalib-only60MHz.py

#python /home/idayan/dataframe2/noisegraph.py

#python /home/idayan/dataframe2/GPHUNTgit.py
#python /home/idayan/dataframe2/GPHUNT202012032122.py

#python /home/idayan/dataframe2/noisegraph.py


python /home/idayan/dataframe2/parallelized-noisegraph.py $1

#python /home/idayan/dataframe2/noisedist-equalarea.py $1

