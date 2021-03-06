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

#import pyds9 as pyd

#from tkp.accessors import sourcefinder_image_from_accessor
#from tkp.accessors import open as open_accessor

from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

import csv

#import tkp.sourcefinder.image
import matplotlib.pyplot as plt
import math
from datetime import datetime

from astropy.coordinates import SkyCoord, match_coordinates_sky,  AltAz, EarthLocation

import tkp.db
import tkp
#import tkp.config

import tkp.db
import logging

from photutils.datasets import make_100gaussians_image
from photutils.aperture import CircularAperture, CircularAnnulus

from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.datasets import make_100gaussians_image

from astropy.stats import sigma_clipped_stats
import pickle


import tkp.db.alchemy
from pandas import DataFrame
#from bokeh.plotting import figure, output_notebook, show
#from bokeh.models import DatetimeTickFormatter


#toavrg202006051431.txt
#toavrg202010030948.txt
#~/toavrg202005181400.txt
with open('/home/idayan/toavrg202005181400.txt','r') as f: # 65805
#with open('/home/idayan/_Toavrg202010030948.txt','r') as f: # 31895
#with open('/home/idayan/toavrg202010030948.txt','r') as f: # 44395 #in aAVErAGED-FINAL somehor half of this date is missing after 10:12
#with open('/home/idayan/toavrg202006051431.txt','r') as f: # 71243
#with open('/home/idayan/toavrg202005121735.txt','r') as f: # wc -l  toavrg202005121735.txt 148906
#with open('/home/idayan/rmngtoavrg202009290730.txt','r') as f: # wc -l rmngtoavrg202009290730.txt /home/idayan/tbavrgd202009290730.txt 102394
#with open('/home/idayan/tbavrgd202009290730.txt','r') as f: # /home/idayan/tbavrgd202009290730.txt 218459
#with open('/home/idayan/imgs202006051431toavrg.txt','r') as f: # two sec seperated imgs; check if they really like that; 87577
#with open('/home/idayan/2-imgsin70-101102.txt','r') as f:
#with open('/home/idayan/imgsin70.txt','r') as f:
    lines=f.read().splitlines()
imagepaths = sorted(lines) 

DTofimagepaths=[]
for i in sorted(imagepaths):

#print i.split('/')[7][:19]
    DTofimagepaths.append(i.split('/')[7][:19])
matching=[]
for i in sorted(set(DTofimagepaths)):
    print 'printing DT of:', i
    matching.append([s for s in imagepaths if "{}".format(i) in s])
#with open('/home/idayan/timestamps_imgsin70.txt','r') as f:
#202010030948
#202005181400
file_name = "/home/idayan/rmng202005181400tmstmpstoavrg.pkl"
#file_name = "/home/idayan/rmng202010030948tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202010030948tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202005121735tmstmpstoavrg.pkl"
#file_name = "/home/idayan/rmng202009290730tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202009290730tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202006051431timestampstoavrg.pkl"
#file_name = "/home/idayan/101102_2timestamps_imgsin70.pkl"
#file_name = "/home/idayan/test_timestamps_imgsin70.pkl"

open_file = open(file_name, "wb")
print 'dumping file...'
pickle.dump(matching, open_file)
#print 'file closing'

open_file.close()

#######  how to read :
####     open_file = open(file_name, "rb")
####     loaded_list = pickle.load(open_file)
####
