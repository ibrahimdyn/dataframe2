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


#with open(path_,'r') as f:
#imgs202006051431toavrg.txt
#rmngtoavrg202006051431.txt


with open('/home/idayan/toAVRG202011080802.txt','r') as f: #100680
#with open('/home/idayan/imgs202006051431toavrg.txt','r') as f: # 68243, in total is:71242
#with open('/home/idayan/imgs202006051431toavrg.txt','r') as f: #87577 imgs202006051431toavrg.txt FIRST FLLWP
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
#202006051431
#202011080802
file_name = "/home/idayan/202006051431tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202006051431tmstmpstoavrg.pkl" #this lastly run for 202011080802 ..pkl file wc -l 210780 /home/idayan/202006051431tmstmpstoavrg.pkl
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

