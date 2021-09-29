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
#dates=[202101040000, 202101040100, 202101040200, 202101031339, 202101031500]
dates=[ 202101031900, 202101031500]
for date in dates:
    
    imglst=glob.glob("/zfs/helios/filer0/mkuiack1/{}/*_all/SB*/imgs/*".format(date))
    N=len(imglst)
    wrr=pd.DataFrame()
    wrr=pd.concat(Parallel(n_jobs=4, backend="multiprocessing", verbose=10)(delayed(llc)(imglst[i]) for i in range(0,N)),ignore_index=True)
    wrr.to_csv('{}.csv'.format(date))
