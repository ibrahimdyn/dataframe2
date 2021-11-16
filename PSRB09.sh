#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 20G
#SBATCH --time 24:00:00
#SBATCH --output=/home/idayan/CALIBRATEDIMAGES/calibration.log



source /home/idayan/env/bin/activate 




#cp /zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits /hddstore/idayan/
#python /home/idayan/dataframe2/crtdataframe.py /home/idayan/imglst/*.fits
#python /home/idayan/dataframe2/crtdataframe.py $SLURM_ARRAY_TASK_ID"/zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits"


#python /home/idayan/dataframe2/crtdataframe.py 
#python /home/idayan/dataframe2/AllDatesDataFrame.py
#python /home/idayan/dataframe2/AllDateDF-Automat.py
#CALIBRATION202008122000.py
python /home/idayan/dataframe2/CALIBRATION202008122000.py
