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

src = [148.56 , 7.66]
def IN60(IMG, source):
    hdl=fits.open(IMG)[0]
    imgcentrecoord = SkyCoord(hdl.header["CRVAL1"],hdl.header["CRVAL2"],unit='deg')
    targetcoord = SkyCoord(source[0],source[1], unit='deg')
    sep = imgcentrecoord.separation(targetcoord).deg
    print sep
    if sep < 69.9:
    #if sep<50.1:
      with open(r'/home/idayan/2-imgsin70-101102.txt', 'a') as f:
      #with open(r'/home/idayan/imgsin50.txt', 'a') as f:
      #with open(r'/home/idayan/imgsin70.txt', 'a') as f:
      #with open(r'/home/idayan/imgsin60-3-10110204.txt', 'a') as f:
                 #f.write(i)
                 f.write(IMG)
                 f.write('\n')
    
    
    return 
  
if __name__ == "__main__":
  
  #if ma
  IN60(image, src)
