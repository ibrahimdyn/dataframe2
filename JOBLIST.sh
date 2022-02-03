#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 15
#SBATCH --mem 20G
#SBATCH --time 420:00:00
#SBATCH --output=/home/idayan/joblist.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))


singularity shell -B /zfs/helios/filer1/mkuiack1/:/opt/Data $HOME/lofar-pipeline.simg

source /opt/lofarsoft/lofarinit.sh


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

for i in 202006040630 202006040830 202006041032 202006041232 202006041433 202006050630 202006051032 202006051232 \
202006051431 202006060227 202006060700  202006061032 202006061232 202006061630 202006061731 202006070309 202006070522 \
202006070700 202006071032 202006071232 202006071501 202006071603 202006081032 202006081232 202006081501 202006081851 \
202006090116 202006090629 202006091032 202006091430 202006091752 202006091853 202006100059 202006201032 202006201232 \
202006201953 202006210153 202006210700 202006230052 202006230105 202006230700 202006231032 202006231232 202006240130 \
202006240700 202006241032 202006241232; do python A12_pipeline/pyscripts/make_joblist.py $i; done

