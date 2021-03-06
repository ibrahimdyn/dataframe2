#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 20G
#SBATCH --time 40:00:00
#SBATCH --output=/home/idayan/testt-callallauto.log
#SBATCH --array=1-10%3
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

echo "your cwd is:" $(pwd)
#for i in glob.glob("/zfs/helios/filer0/idayan/Calimgs/*.fits"):
#b=`ls /zfs/helios/filer0/idayan/Calimgs/*.fits`
b=`ls /zfs/helios/filer1/idayan/$1/*_all/SB*/imgs/*.fits`

echo "this files will be processed:"
for i in "$b"; do echo "$i"; done
echo "processed files' end"

#c=`ls /zfs/helios/filer1/idayan/202006040630/*_all/SB*/imgs/*.fits`
c=`ls /zfs/helios/filer1/idayan/$1/*_all/SB*/imgs/*.fits`
ls $c > test44.txt
echo "echoing sed command; ready steady go:"
ls `sed $SLURM_ARRAY_TASK_ID'q;d' test44.txt`
echo "echoED sed command!!!"
python /home/idayan/dataframe2/TESTcalautomate.py `sed $SLURM_ARRAY_TASK_ID'q;d' test44.txt` 
#python /home/idayan/dataframe2/Cal-All-automate.py `sed $SLURM_ARRAY_TASK_ID'q;d' $test33.txt` 
#/home/mkuiack1/A12_pipeline/helios_pipeline/A12_pipelinearray.sh `sed $SLURM_ARRAY_TASK_ID'q;d' $OBSSLICEFILE` 
#for i in $b;do python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$i; done
#python /home/idayan/dataframe2/Cal-All-automate.py --indir=/zfs/helios/filer0/idayan/Calimgs/ --fitsfile="$i"

#srun hostname
#FILES=(assembled_reads/*.sorted.bam)    
#FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

#FILES=(assembled_reads/*.sorted.bam)    
#FILE=${$b[$SLURM_ARRAY_TASK_ID]}
#FILE=${($b)[$SLURM_ARRAY_TASK_ID]}
#echo $SLURM_ARRAY_TASK_ID
#FILE=${"$b"[$SLURM_ARRAY_TASK_ID]}
#echo $FILE
