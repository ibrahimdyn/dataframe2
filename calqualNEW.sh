#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 10
#SBATCH --mem 40G
#SBATCH --time 240:00:00
#### ###SBATCH --array=13-17
#SBATCH --array=1-320%50
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
#ls /zfs/helios/filer1/idayan/CALed/202006040830/2*.fits > ~/filEE.txt
#ls /zfs/helios/filer1/idayan/CALed/202009290730/2*.fits > ~/file202009290730.txt
#echo "echoing head of tthe file"
#head filee.txt
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
    #~/filEE.txt
    #319402 /home/idayan/toavrg202009290730.txt # 319402
    #tocalqualNEW202010030948.txt
    echo "0 getting LINE"
    
    LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202005121735.txt) # check first withoutt beam correct
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202006041232.txt) # 31k rows std is 0.526 without beam correction (check again whether 1.67 is true) ! nope it is 1
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202006041232.txt) # 56528 tocalqualNEW202006041232.txt !!! cal qual result of 19k images 1.67
    echo "1 got the LINE"
    # now i will try same 0604 without beam correction
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202010030948.txt) # 11399 tocalqualNEW202010030948.txt !!! cal qual result of 8735 images 0.35
    #LINE=$(sed -n "$N"p /home/idayan/toavrg202009290730.txt) # 319402
    #LINE=$(sed -n "$N"p /home/idayan/file202009290730.txt) 
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
    python /home/idayan/dataframe2/calqualNEW3.py $LINE
    #python /home/idayan/dataframe2/calqualNEW2.py $LINE
    #python /home/idayan/dataframe2/calqualNEW.py $LINE
    #python /home/idayan/dataframe2/CALQUAL5.py $LINE
    #python /home/idayan/dataframe2/CALQUALpar.py $LINE
    
    #python /home/idayan/dataframe2/testconfautomate.py --fitsfile=$LINE
    echo "after processing, NoFiles"
    #echo $(wc -l ~/ALL202007Dates2.txt)
    
done



#echo "this files will be processed:"
#for i in "$b"; do echo "$i"; done
#echo "processed files' end"
#for i in $b;do python /home/idayan/dataframe2/Cal-All-automate.py --fitsfile=$i; done
#python /home/idayan/dataframe2/Cal-All-automate.py --indir=/zfs/helios/filer0/idayan/Calimgs/ --fitsfile="$i"
