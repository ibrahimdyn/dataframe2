#!/bin/bash 
##SBATCH -N 1
##SBATCH --ntasks-per-node 1
##SBATCH --cpus-per-task 12
##SBATCH --mem 20G
##SBATCH --time 12:00:00
##SBATCH --array=1-128%32

source /home/idayan/env/bin/activate 

slurm_job_id=os.environ['SLURM_ARRAY_TASK_ID']
folders=(glob.glob('/zfs/helios/filer0/mkuiack1/202008122000/*_all'))


#cp /zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits /hddstore/idayan/
#python /home/idayan/dataframe2/crtdataframe.py /home/idayan/imglst/*.fits
#python /home/idayan/dataframe2/crtdataframe.py $SLURM_ARRAY_TASK_ID"/zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits"


python /home/idayan/dataframe2/crtdataframe.py 
