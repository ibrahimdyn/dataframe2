#!/bin/bash 
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 6G
#SBATCH --time 240:00:00
#### ###SBATCH --array=13-17
#SBATCH --array=1-30%30
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

echo "getting date variable"
DATE = $1
echo "printing date variable"
echo $DATE



START=$SLURM_ARRAY_TASK_ID
NUMLINES=30
#NUMLINES=550
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
    #56504 tocalqualNEW-202010030948.txt
    #~/ALL-TXT/tocalqualNEW-202102181807.txt
    #~/ALL-TXT/tocalqualNEW-202010201005.txt ; done
    
    #56504 /home/idayan/ALL-TXT/tocalqualNEW-202010201130.txt
    #~/ALL-TXT/tocalqualNEW-202010060710.txt
    # -bash-4.2$ wc -l  ~/ALL-TXT/tocalqualNEW-202011080802.txt
    #56622 
    #56504 /home/idayan/ALL-TXT/tocalqualNEW-202011161001.txt
    # ~/ALL-TXT/tocalqualNEW-202006041433.txt
    #  /home/idayan/ALL-TXT/tocalqualNEW-202007231100.txt
     
    #tocalqualNEW-202005181400-REMAINING-.txt
    # /home/idayan/ALL-TXT/tocalqualNEW-202009290730.txt
    
    #~/ALL-TXT/tocalqualNEW-202012140600.txt
    
    #/home/idayan/ALL-TXT/tocalqualNEW-202101031339.txt
    LINE=$(sed -n "$N"p $1) # 
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202012131700.txt) # 
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202012131600.txt) # 50968
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202012140600.txt) # 51341
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202102230145.txt) # 60384
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202102230245.txt) # 60400
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202009290730.txt) # 316366
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202009280800.txt) # 147641
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202005291110.txt) # 226474
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202005192025.txt) #42463
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202005181400-REMAINING-.txt) #54k - remaining
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202005181400.txt) #141552
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202005131530.txt) # 104360
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/toucal-202005121735-.txt) # 81531
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/toLIMITEDcalqualNEW-202005052000-.txt) # 86000 from 180k
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010271137.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010201417.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202007231100.txt) # 83058
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010201235.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202006041433.txt) # 28252
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010030719.txt) # 28252
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010191408.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010030719.txt)  # 42415 ---- again 202010030719 !
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010191408.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010172155.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010030719.txt) # 42415 ---- again 202010030719 !
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202011270912.txt) # 56504 
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202006201232.txt) # 707781
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202006231232.txt) # 70781
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202006061731.txt) # 28308
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202006061630.txt) # 14163
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202011161001.txt) # 56504
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202011100907.txt) # 56504
    
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202006041232.txt) # 56622
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/fluxcal-remaining202011080802.txt) # 89496
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202011080802.txt) # 169494
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202006061232.txt) # 70769
    #LINE=$(sed -n "$N"p  /home/idayan/ALL-TXT/tocalqualNEW-202011021014.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202006051431.txt) # 53759
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010131021.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010060710.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010031250.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010201130.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010201005.txt) # 55436
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202102181807.txt) # 11487
    #LINE=$(sed -n "$N"p /home/idayan/ALL-TXT/tocalqualNEW-202010131201.txt) # 56504
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW-202010031118.txt) # 68024 tocalqualNEW-202010031118.txt
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW-202010030948.txt) # 56504 tocalqualNEW-202010030948.txt
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202005181400.txt) # 141552 202005181400 tocalqualNEW202005181400
    #LINE=$(sed -n "$N"p /home/idayan/imgpathstofluxcal202009290730.txt) # 254798 imgpathstofluxcal202009290730.txt
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202006051232.txt) # 56621 
    #LINE=$(sed -n "$N"p /home/idayan/tocalqualNEW202006041232.txt) # 56528 tocalqualNEW202006041232.txt # wtih 0.45 std only 1 percent pass the filter
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

