#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 15
#SBATCH --mem 20G
#SBATCH --time 420:00:00
#SBATCH --output=/home/idayan/calw2refnewrad202009240800.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))








#source /home/idayan/env/bin/activate
source /home/idayan/new_env/bin/activate




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
#python /home/idayan/dataframe2/caw2ref-202012032122.py 
#python /home/idayan/dataframe2/calw2ref202012132000inner.py
python /home/idayan/dataframe2/calw2ref-202009240800.py

