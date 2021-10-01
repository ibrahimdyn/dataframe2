import sys
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
import time


#imglst=glob.glob("/home/idayan/imglst/*.fits")

#len(imglst)

#start= time.time()
#imglst = glob.glob(sys.argv[1])
#imglst[0]

#imglst=glob.glob("/zfs/helios/filer0/mkuiack1/202101040500/*_all/SB*/imgs/*") ## 1.
#imglst=glob.glob("/zfs/helios/filer0/mkuiack1/202101040400/*_all/SB*/imgs/*") ## 2.
#imglst=glob.glob("/zfs/helios/filer0/mkuiack1/202101040300/*_all/SB*/imgs/*") ## 3.


#imglst=glob.glob("/hddstore/idayan/*.fits")
#imglst=glob.glob("/home/idayan/imglst/*.fits")
#print(len(imglst))


#z=SkyCoord.from_name("PSR J1231-1411")
#z=SkyCoord.from_name("4C 14.27") #
z=SkyCoord.from_name("PSRB0950+08")
def llc(filee):

    hdulist= (fits.open(filee))[0]

    w=wcs.WCS(hdulist.header) #
    #wcoo=w.wcs_world2pix(231.794534, 51.100721 ,1,1,1)
    #wcoo=w.wcs_world2pix()

    ### wcoo=w.wcs_world2pix(100, 100 ,1,1,1)  ## neww.csv came from here
    wcoo=w.wcs_world2pix(z.ra, z.dec,1,1,1)

    arrwcoo=np.asarray(wcoo)  #
   # aa=pd.DataFrame(columns =['I', 'Freq', 'Time'])
   # aa=aa.append({'I':(hdulist.data)[0,0,int(arrwcoo[0]),int(arrwcoo[1])],

   #               'Freq':hdulist.header["RESTFRQ"], 'Time':hdulist.header["DATE-OBS"][17:23]}, ignore_index=True)

    aa=pd.DataFrame(columns =['I', 'Freq', 'Time'])
    aa=aa.append({'I':(hdulist.data)[0,0,int(arrwcoo[0]),int(arrwcoo[1])],
            'Freq':hdulist.header["CRVAL3"], 'Time':hdulist.header["DATE-OBS"][11:23]}, ignore_index=True)
    return aa

##%%timeit

#dates=[202101032000, 202101031700]
dates=[202012131600, 202012131700, 20201213180, 202012131900, 202012132000, 202012132100, 202012132200, 202012132300, 
       202012140000, 202012140100, 202012140200, 202012140300, 202012140400, 202012140500, 202012140600, 
       202012180840, 202012181242,
       202012190851, 202012191033 ,202012191135, 202012191236, 
       202012200847,
       202012220815, 202012221815,
       202012230217, 202012230741, 202012230917, 202012231147, 202012231340]
for date in dates:
    
    imglst=glob.glob("/zfs/helios/filer0/mkuiack1/{}/*_all/SB*/imgs/*".format(date))
    N=len(imglst)
    wrr=pd.DataFrame()
    #while arcwoo is not None
    try:
        
        
        wrr=pd.concat(Parallel(n_jobs=4, backend="multiprocessing", verbose=10)(delayed(llc)(imglst[i]) for i in range(0,N)),ignore_index=True)
        wrr.to_csv('Dateof{}.csv'.format(date))
        #pass
    #if arcwoo is None:
    #    pass to second argument
        #wrr.to_csv('Dateof{}.csv'.format(date))
    except Exception:
        print(date)
        print('it gave error')
        continue
        #print('An exception occureddd')
    #else:
        #wrr.to_csv('Dateof{}.csv'.format(date))
