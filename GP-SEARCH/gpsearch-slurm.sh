#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 40G
#SBATCH --time 240:00:00
##### ###SBATCH --array=13-17
#SBATCH --array=1-400%100
### ###SBATCH --exclude=helios-cn[016-020]
#### #SBATCH --output=/home/idayan/TESTCALaut.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))


#source /home/idayan/env/bin/activate
source /home/idayan/new_env/bin/activate

echo "your cwd is:" $(pwd)
#for i in glob.glob("/zfs/helios/filer0/idayan/Calimgs/*.fits"):
#b=`ls /zfs/helios/filer0/idayan/Calimgs/*.fits`
#b=`ls /zfs/helios/filer1/idayan/$1/*_all/SB*/imgs/*.fits`

#echo "this files will be processed:"
#for i in "$b"; do echo "$i"; done
#echo "processed files' end"

#c=`ls /zfs/helios/filer1/idayan/202006040630/*_all/SB*/imgs/*.fits`
#c=`ls /zfs/helios/filer1/idayan/$1/*_all/SB*/imgs/*.fits`
#ls $c > ~/test44.txt
echo "echoing sed command; ready steady go:"
#ls `sed $SLURM_ARRAY_TASK_ID'q;d' test44.txt`
echo "echoED sed command!!!"
#python /home/idayan/dataframe2/TESTcalautomate.py `sed $SLURM_ARRAY_TASK_ID'q;d' test44.txt` 
#Cal-All-automate.py
#python /home/idayan/dataframe2/Cal-All-automate.py `sed $SLURM_ARRAY_TASK_ID'q;d' ~/test44.txt` 

#--fitsfile="$i"
#python /home/idayan/dataframe2/Cal-All-automate.py  --fitsfile=`sed $SLURM_ARRAY_TASK_ID'q;d' ~/test44.txt`


START=$SLURM_ARRAY_TASK_ID
NUMLINES=1000
STOP=$((SLURM_ARRAY_TASK_ID*NUMLINES))
START="$(($STOP - $(($NUMLINES - 1))))"
#START="$(($STOP - $(($NUMLINES - 1))))"

echo "START=$START"
echo "STOP=$STOP"

for (( N = $START; N <= $STOP; N++ ))
do
    echo $N
    #LINE=$(sed -n "$N"p ~/3Dates.txt)
    #ALL3Dates
    
    #LINE=$(sed -n "$N"p ~/imgsin60-3-10110204.txt)
    LINE=$(sed -n "$N"p ~/imgsin60.txt)
    #LINE=$(sed -n "$N"p ~/ALL202007Dates2.txt)
    echo $LINE
    #echo "before processing, NoFiles"
    #echo $(wc -l ~/ALL3Dates.txt)
    #echo $(wc -l ~/imgsin60.txt)
    #echo $(wc -l ~/mgsin60-3-10110204.txt)
    
    #echo $(wc -l ~/imgsin60-3-10110204.txt)
    
    python /home/idayan/dataframe2/GP-SEARCH/GPsearch-un202007.py $LINE
    #python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$LINE
    
    #python /home/idayan/dataframe2/testconfautomate.py --fitsfile=$LINE
    echo "processing done"
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
done



#echo "this files will be processed:"
#for i in "$b"; do echo "$i"; done
#echo "processed files' end"
#for i in $b;do python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$i; done
#python /home/idayan/dataframe2/Cal-All-automate.py --indir=/zfs/helios/filer0/idayan/Calimgs/ --fitsfile="$i"


