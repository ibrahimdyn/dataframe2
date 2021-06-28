import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import glob
#from mpl_toolkits import mplot3d
import astropy.wcs as wcs
from astropy.table import Table
import pandas as pd
from astropy.coordinates import SkyCoord 
from astropy.coordinates import Angle
from astropy import units as u
import matplotlib as mpl
#import joblib
from joblib import Parallel, delayed 
import concurrent
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Process
from multiprocessing import Process
import io
import time


imglst=glob.glob("/home/idayan/imglst/*")
#imglst=glob.glob("/zfs/helios/filer0/mkuiack1/202008122000/*_all/SB*/imgs/*")   


N=len(imglst)
#z=SkyCoord.from_name("PSR J1231-1411")
#z=SkyCoord.from_name("4C 14.27") #
def llc(filee):

    hdulist= (fits.open(filee))[0]     
    
    w=wcs.WCS(hdulist.header) #
    #wcoo=w.wcs_world2pix(231.794534, 51.100721 ,1,1,1) 
    #wcoo=w.wcs_world2pix()
    wcoo=w.wcs_world2pix(100, 100 ,1,1,1)  ##
    arrwcoo=np.asarray(wcoo)  #
   # aa=pd.DataFrame(columns =['I', 'Freq', 'Time'])
   # aa=aa.append({'I':(hdulist.data)[0,0,int(arrwcoo[0]),int(arrwcoo[1])], 
    
   #               'Freq':hdulist.header["RESTFRQ"], 'Time':hdulist.header["DATE-OBS"][17:23]}, ignore_index=True)
    
    aa=pd.DataFrame(columns =['I', 'Freq', 'Time'])
    aa=aa.append({'I':(hdulist.data)[0,0,int(arrwcoo[0]),int(arrwcoo[1])], 
            'Freq':hdulist.header["CRVAL3"], 'Time':hdulist.header["DATE-OBS"][17:23]}, ignore_index=True)
    
    #bbwrr=pd.DataFrame()
    #bbwrr=pd.concat(aa,ignore_index=True)
    
    
    return aa
  
#def llc2(x):
#wrr=pd.DataFrame() 
#wrr=pd.concat((llc(imglst[i]) for i in range(16000,16400)),ignore_index=True) 



#    print(f'Sleeping {seconds} second(s)...')
#    time.sleep(seconds)
#    return f'Done Sleeping...{seconds}'
start= time.time()
#%time
#futures=[]
wrr=pd.DataFrame()
with concurrent.futures.ProcessPoolExecutor() as executor:
    
    #secs= glob.glob(("/home/idayan/imglst/*"))
    #secs = [5, 4, 3, 2, 1]
    #executor.map(llc, imglst)
 #   results = executor.map(llc, imglst)
    
    #wrr=pd.DataFrame()
    #for result in results:
    #    futures.append(result)
    #future  = executor.submit(llc,imglst)
    #futures.append(future)
    

#for number in couple_ods:
#future=executor.submit(task,number)
#futures.append(future)

    for result in executor.map(llc, imglst):
        #dostuff(result)
        wrr=wrr.append(result)
        #wrr.to_pickle('DF.pkl')
        wrr.to_csv("/home/idayan/DFcsv.csv",index=True)
        
#wrr.to_csv("/home/idayan/weeee.txt")
#wrr.to_csv("/home/idayan/weeee.t")
#OBS_JOBS.to_csv("$HOME/{}_JOBS.txt".format(obs), index=False, header=False, sep=" ")
#wrr.to_pickle('rerer.pkl')
end= time.time()

print(end-start)
    
