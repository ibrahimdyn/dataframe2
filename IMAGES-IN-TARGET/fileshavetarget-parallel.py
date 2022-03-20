import os
import sys
import csv

from datetime import datetime
#import datetime

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
#import pyds9 as pyd
from tkp.accessors import sourcefinder_image_from_accessor
from tkp.accessors import open as open_accessor


from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

import csv

import tkp.sourcefinder.image
import matplotlib.pyplot as plt
import math

from astropy.coordinates import SkyCoord, match_coordinates_sky,  AltAz, EarthLocation

image=sys.argv[1]
print "starting fileshavetarget-parallel.py "
print "prinnting img:"
print image

#folderlist=sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/2*"))
#folderlist=sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202010*")) \
#+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202011*")) \
#+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202102*")) \
#+ sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202104*"))


CS002 = EarthLocation.from_geocentric (3826577.109500000, 461022.900196000, 5064892.758, 'm')
position = SkyCoord(148.56*u.degree, 
                    7.66*u.degree)

check_list=[]
in60=[]
out60=[]

#for j in folderlist:
#    globbed=glob.glob("{}/*.fits".format(j))
#    print("printing folder")
#    print(j)
  
    #for i in globbed:
def IN60(IM): 
  #hdl=fits.open(i)[0]
  hdl=fits.open(IM)[0]
  obstime_=hdl.header["DATE-OBS"]
  altaz=position.transform_to(AltAz(obstime=obstime_, location=CS002))
  #altaz=position.transform_to(AltAz(obstime=[j for j in obs_times], location=CS002))
  imgcentercoord=SkyCoord(hdl.header["CRVAL1"],hdl.header["CRVAL2"],unit="deg")
  targetcoord=SkyCoord(altaz.az.deg,altaz.alt.deg,unit="deg")
  sep=imgcentercoord.separation(targetcoord).deg
  #obs_list_t=[]
  #check_list.append(i)
  #if sep < 20:
  #    in60.append(i)
  #    print i
  #else:
  #    out60.append(i)

  #print sep
  if sep < 60.9:
      #print i
      print IM
      with open(r'/home/idayan/imgsin60-2-10110204.txt', 'a') as f:
                 #f.write(i)
                 f.write(IM)
                 f.write('\n')
                 #writer = csv.writer(f)
                 #writer.writerow(i)
#ssss
if __name__ == "__main__":
  
  #if ma
  IN60(image)
