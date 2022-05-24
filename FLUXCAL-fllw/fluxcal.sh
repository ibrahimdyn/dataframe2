#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 40G
#SBATCH --time 240:00:00
#SBATCH --array=1-255%50



#source /home/idayan/env/bin/activate
source /home/idayan/new_env/bin/activate

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
    #/home/idayan/imgs202006051431toavrg.txt
    #Send0_rmngimgstofluxcal202010030948.txt
    #ALL202011080802tofluxcal.txt
    LINE=$(sed -n "$N"p /home/idayan/ALL202011080802tofluxcal.txt)  #153077
    #LINE=$(sed -n "$N"p /home/idayan/Send0_rmngimgstofluxcal202010030948.txt) 
    
    #LINE=$(sed -n "$N"p /home/idayan/imgs202006051431toavrg.txt) # must be "to fluxcal"
    #LINE=$(sed -n "$N"p /home/idayan/imgpathstofluxcal202009290730.txt)
    #LINE=$(sed -n "$N"p /home/idayan/202012in70toCAL.txt)
    #LINE=$(sed -n "$N"p ~/REMAINIGDATES.txt)
    #LINE=$(sed -n "$N"p ~/ALL202007Dates2.txt)
    
    echo $LINE
    echo "before processing, NoFiles"
    #echo $(wc -l ~/ALL3Dates.txt)
    #echo $(wc -l ~/REMAINIGDATES.txt)
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
    python /home/idayan/dataframe2/FLUXCAL-fllw/fluxcal.sh --fitsfile=$LINE
    #python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$LINE
    
    #python /home/idayan/dataframe2/testconfautomate.py --fitsfile=$LINE
    echo "after processing, NoFiles"
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
done

