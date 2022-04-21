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
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import DatetimeTickFormatter

logging.basicConfig(level=logging.INFO)
output_notebook()

query_loglevel = logging.WARNING 





with open('/home/idayan/imgsin70.txt','r') as f:
    lines=f.read().splitlines()
imagepaths = sorted(lines)   

DTofimagepaths=[]

for i in sorted(imagepaths):
#for i in sorted(imagepaths[21000:21600]):
    #print i.split('/')[7][:19]
    DTofimagepaths.append(i.split('/')[7][:19])

matching=[]
for i in sorted(set(DTofimagepaths)):
    #print i
    matching.append([s for s in imagepaths if "{}".format(i) in s])



for k in matching:
    
    #for j in k:
    #for j in keylist:
    #print 'this is j:', j
    myfiles=k
    print 'myfiles', k
    if len(myfiles)>11:
        
        #myfiles=DF[DF['File_Path'].str.contains(r'{}(?!$)'.format(j))]['File_Path']
        #myfiles=DF[DF['File_Path'].str.contains(r'2020-06-04T08:32:02(?!$)')]['File_Path']

        #nofile=len(DF[DF['File_Path'].str.contains(r'2020-06-04T08:32:02(?!$)')]['File_Path'])
        nofile =len(k)
        print 'nofile', nofile
        s=0
        data_1=np.zeros((2300,2300))
        data_2=np.zeros((2300,2300))
        freqlist1=[]
        freqlist2=[]
        print 'this is =my files :', myfiles


        for i in range(len(myfiles)):

            print i
            #print myfiles[i]
            #fitsimg = fits.open(myfiles[i])[0]
            #fitsimg = fits.open(cfg.indir+cfg.fitsfile)[0]

            if s < nofile/2:
            #while s < nofile:
                print s,'inside first'
                data_1 += fits.getdata(myfiles[i],
                                          header=False)[0,0,:,:]
                #data_1 = data_1/(nofile/2)
                if nofile % 2 == 0:
                    data_1 = data_1/((nofile/2))
                  
                if nofile % 2 == 1:
                    data_1 = data_1/(math.floor(nofile/2+1))

                freqlist1.append(float(myfiles[i].split('-')[-2][1:6]))
                #np.mean(freqlist1)*0.195

                #s += 1

            if s>=nofile/2:

                print 'second if', s
                data_2 += fits.getdata(myfiles[i],
                                          header=False)[0,0,:,:]
                #data_2 = data_2/(nofile/2)
                if nofile % 2 == 0:
                    data_2 = data_2/((nofile/2))
                  
                if nofile % 2 == 1:
                    data_2 = data_2/(math.floor(nofile/2))

                freqlist2.append(float(myfiles[i].split('-')[-2][1:6]))
                #s += 1
                print 'end if', s


            s+=1  

        print 'this is i:',i

        filenum1=int(math.floor(nofile/2.-1))
        filenum2=int(math.floor(nofile/2.+1))

        DIR=myfiles[filenum1].split('/')[6]
        #DIR=myfiles[int(math.floor(nofile/2.-1))].split('/')[6]
        #DIR=myfiles[math.floor(nofile/2.-1)].split('/')[6]

        #valuefreqlst1=round(np.mean(freqlist1)*0.195,3)
        #valuefreqlst2=round(np.mean(freqlist2)*0.195,3)
        valuefreqlst1=np.mean(freqlist1)*0.195
        valuefreqlst2=np.mean(freqlist2)*0.195


        fitsimg1 = fits.open(myfiles[filenum1])[0]
        fitsimg2 = fits.open(myfiles[filenum2])[0]
        #fitsimg1 = fits.open(myfiles[math.floor(nofile/2.-1)])[0]
        #fitsimg2 = fits.open(myfiles[math.floor(nofile/2.+1)])[0]
        fitsimg1.data=data_1
        fitsimg2.data=data_2


        fits.setval(myfiles[filenum1],'CRVAL3', value=valuefreqlst1)
        fits.setval(myfiles[filenum1],'RESTFRQ', value=valuefreqlst1)
        fits.setval(myfiles[filenum1],'RESTFREQ', value=valuefreqlst1)


        #fits.setval(myfiles[math.floor(nofile/2.-1)],'CRVAL3', value=valuefreqlst1)
        #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFRQ', value=valuefreqlst1)
        #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFREQ', value=valuefreqlst1)

        fits.setval(myfiles[filenum2],'CRVAL3', value=valuefreqlst2)
        fits.setval(myfiles[filenum2],'RESTFRQ', value=valuefreqlst2)
        fits.setval(myfiles[filenum2],'RESTFREQ', value=valuefreqlst2)


        #fits.setval(myfiles[math.floor(nofile/2.+1)],'CRVAL3', value=valuefreqlst2)
        #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFRQ', value=valuefreqlst2)
        #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFREQ', value=valuefreqlst2)

        #myfiles[math.floor(nofile/2-1)].split('/')[-1].split('-')[3]
        #myfiles[math.floor(nofile/2+1)].split('/')[-1].split('-')[3]
        #myfiles[index].split('/')[-1].split('-')[3]

        datetime1=myfiles[filenum1].split('/')[-1][:19]
        datetime2=myfiles[filenum2].split('/')[-1][:19]
        #datetime1=myfiles[math.floor(nofile/2.-1)].split('/')[-1][:19]
        #datetime2=myfiles[math.floor(nofile/2.+1)].split('/')[-1][:19]

        #getSB1=int(np.floor((1024./200.)*valuefreqlst1))
        #getSB2=int(np.floor((1024./200.)*valuefreqlst2))

        notgetSB1=round(valuefreqlst1,0)
        notgetSB2=round(valuefreqlst2,0)


        filename1= '%s.fits' % (datetime1 + "-S" + str(notgetSB1))
        filename2= '%s.fits' % (datetime2 + "-S" + str(notgetSB2))

        _path='/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR)

        if os.path.exists(_path):
            #print 1   
            fitsimg1.writeto('/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR)+ '/' + filename1, overwrite=True)
            fitsimg2.writeto('/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR)+ '/' + filename2, overwrite=True)
        if not os.path.exists(_path):
            os.makedirs(_path)
            fitsimg1.writeto('/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR)+ '/' + filename1, overwrite=True)
            fitsimg2.writeto('/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR) +'/' + filename2, overwrite=True)

