import os
import sys
import csv

from datetime import datetime
import argparse
import numpy as np
import pandas as pd

#from sourcefinder.accessors import open as open_accessor
#from sourcefinder.accessors import sourcefinder_image_from_accessor

from tkp.accessors import sourcefinder_image_from_accessor
from tkp.accessors import open as open_accessor

from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

import glob
from astropy.coordinates import SkyCoord
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
import pyds9 as pyd
from tkp.accessors import sourcefinder_image_from_accessor
from tkp.accessors import open as open_accessor


from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

import csv

import tkp.sourcefinder.image
import matplotlib.pyplot as plt
import math
from datetime import datetime

from astropy.coordinates import SkyCoord, match_coordinates_sky,  AltAz, EarthLocation


obsdir=sorted(glob.glob("/zfs/helios/filer0/mkuiack1/*"))

SROBSDIR=obsdir[:66]

#imglst=[]
fits_list= sorted(glob.glob("/zfs/helios/filer0/mkuiack1/202009290730/*_all/SB*/imgs/*fits"))
for i in SROBSDIR[5:]:
    #imglst = sorted(glob.glob(i+ "/*_all"))
    #np.append(fits_list, )
    fits_list.append(sorted(glob.glob(i+ "/*_all/SB*/imgs/*fits")))
    
    #print i
    #print imglst
    

CS002 = EarthLocation.from_geocentric (3826577.109500000, 461022.900196000, 5064892.758, 'm')
position = SkyCoord(148.56*u.degree, 
                    7.66*u.degree)

arrfits_list=fits_list[0][:]
notin50 = []
for i in arrfits_list:
    hdl=fits.open(i)[0]
    obstime_=hdl.header["DATE-OBS"]
    altaz=position.transform_to(AltAz(obstime=obstime_, location=CS002))
    #altaz=position.transform_to(AltAz(obstime=[j for j in obs_times], location=CS002))
    imgcentercoord=SkyCoord(hdl.header["CRVAL1"],hdl.header["CRVAL2"],unit="deg")
    targetcoord=SkyCoord(altaz.az.deg,altaz.alt.deg,unit="deg")
    sep=imgcentercoord.separation(targetcoord).deg
    
    if sep> 50:
        notin50.append(i)
        
    #obs_list_t=[]
    #if sep < 75:
    #    obs_list_t.append(i)
    #    print i 
with open('Notin50_file.csv', 'w') as f:
    for item in notin50:
        f.write("%s\n" % item)
   
