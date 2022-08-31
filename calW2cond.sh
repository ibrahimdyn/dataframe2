#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 40G
#SBATCH --time 240:00:00
#### ###SBATCH --array=13-17
#SBATCH --array=1-200%90
### #SBATCH --array=1-650%100
### #SBATCH --exclude=helios-cn[016-020]
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
#NUMLINES=1000
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
    #/home/idayan
    
    #ALLimgpathstofluxcal202009290730.txt
    #ALLimgpathstofluxcal-202005121735.txt
    #ALLimgpathstofluxcal-202005052000.txt
    #ALLimgpathstofluxcal1-202005052000.txt
    #_intarALLimgpathstofluxcal1-202005052000.txt
    #tocalqualNEW202006041232.txt
    
    LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202006041232.txt) # 56528 tocalqualNEW202006041232.txt # wtih 0.45 std only 1 percent pass the filter
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202005121735UP1.txt) #198167
    
    #LINE=$(sed -n "$N"p /home/idayan/_intarALLimgpathstofluxcal1-202005052000.txt) # 89000
    #LINE=$(sed -n "$N"p /home/idayan/ALLimgpathstofluxcal1-202005052000.txt) #193222
    #LINE=$(sed -n "$N"p /home/idayan/ALLimgpathstofluxcal1-202005181400.txt)  # 141555
    #LINE=$(sed -n "$N"p /home/idayan/ALL202005052000tofluxcal.txt)  # 45287
    #LINE=$(sed -n "$N"p /home/idayan/ALL202011080802tofluxcal.txt)  #153077
    #LINE=$(sed -n "$N"p /home/idayan/ALLimgpathstofluxcal-202005121735.txt) # 202785
    #LINE=$(sed -n "$N"p /home/idayan/ALLimgpathstofluxcal202009290730.txt) #remaining of 202009290730 155288 
    #LINE=$(sed -n "$N"p /home/idayan/imgpathstofluxcal202009290730.txt)
    #LINE=$(sed -n "$N"p /home/idayan/202012in70toCAL.txt)
    #LINE=$(sed -n "$N"p ~/REMAINIGDATES.txt)
    #LINE=$(sed -n "$N"p ~/ALL202007Dates2.txt)
    
    echo $LINE
    echo "before processing, NoFiles"
    #echo $(wc -l ~/ALL3Dates.txt)
    #echo $(wc -l ~/REMAINIGDATES.txt)
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
    
    python /home/idayan/dataframe2/calW2cond.py --fitsfile=$LINE
    #python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$LINE
    
    #python /home/idayan/dataframe2/testconfautomate.py --fitsfile=$LINE
    echo "after processing, NoFiles"
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
done



#echo "this files will be processed:"
#for i in "$b"; do echo "$i"; done
#echo "processed files' end"
#for i in $b;do python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$i; done
#python /home/idayan/dataframe2/Cal-All-automate.py --indir=/zfs/helios/filer0/idayan/Calimgs/ --fitsfile="$i"

