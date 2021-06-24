#!/bin/bash 

source /home/idayan/env/bin/activate 

#python /home/idayan/dataframe2/crtdataframe.py /home/idayan/imglst/*.fits
python /home/idayan/dataframe2/crtdataframe.py /zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*.fits
