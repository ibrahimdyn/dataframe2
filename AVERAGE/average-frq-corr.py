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

#logging.basicConfig(level=logging.INFO)
#output_notebook()

#query_loglevel = logging.WARNING 


#with open('/home/idayan/imgsin70.txt','r') as f:
#    lines=f.read().splitlines()
#imagepaths = sorted(lines)   

#DTofimagepaths=[]
#for i in sorted(imagepaths):
    #print i.split('/')[7][:19]
#    DTofimagepaths.append(i.split('/')[7][:19])

#matching=[]
#for i in sorted(set(DTofimagepaths)):
    #print i
#    matching.append([s for s in imagepaths if "{}".format(i) in s])
#/zfs/helios/filer1/idayan/fllwpDATES/AVERAGED/
file_name = "/home/idayan/TMSTMPStoavrg-ucaledimgs-202010031250.pkl"  # 7537
#file_name = "/home/idayan/TMSTMPStoavrg-ucaledimgs-202010201005.pkl" # 35647
#file_name = "/home/idayan/TMSTMPStoavrg-ucaledimgs-202010031118.pkl" # 37824
#file_name = "/home/idayan/TMSTMPStoavrg-ucaledimgs-202010030948.pkl" # 39360
#file_name = "/home/idayan/tmstmpstoavrg-ucaledimgs-202010030948.pkl" # 39360
#file_name = "/home/idayan/UCALED202009290730tmstmpstoavrg.pkl" #130578 /home/idayan/TOtimestamp202005121735.txt
#file_name = "/home/idayan/UCALED202005121735tmstmpstoavrg.pkl" #81531 /home/idayan/TOtimestamp202005121735.txt
#file_name = "/home/idayan/rmng202005181400tmstmpstoavrg.pkl" #65k from 180k ?
#file_name = "/home/idayan/rmng202010030948tmstmpstoavrg.pkl"
#file_name = "/home/idayan/rmngctmstmpstoavrg.pkl" #rmng of mng202010030948 !!! wrong
#file_name = "/home/idayan/202010030948tmstmpstoavrg.pkl" #70k
#file_name = "home/idayan/202006051431tmstmpstoavrg.pkl" #70k
#file_name = "/home/idayan/202011080802tmstmpstoavrg.pkl" #210780 comingfrom with open('/home/idayan/toAVRG202011080802.txt','r') as f: #100680
#file_name = "/home/idayan/202005121735tmstmpstoavrg.pkl" 
#file_name = "/home/idayan/rmng202009290730tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202009290730tmstmpstoavrg.pkl"
#file_name = "/home/idayan/202006051431timestampstoavrg.pkl" # this one stayed bcs of 2 sec issue
#file_name = "/home/idayan/101102_2timestamps_imgsin70.pkl"
#file_name = "/home/idayan/test_timestamps_imgsin70.pkl"
open_file = open(file_name, "rb")
#loaded_list = pickle.load(open_file)
matching = pickle.load(open_file)



set1_subbands=['S281', 'S284', 'S287', 'S291', 'S294', 'S298', 'S301', 'S304']
set2_subbands=['S308', 'S311', 'S315', 'S318', 'S321', 'S325', 'S328', 'S332']


def img_averager(list_sametimestamps):    
    if len(list_sametimestamps)>7:    
        set1=[]
        set2=[]
        myfiles = list_sametimestamps
        #nofile  = len(list_sametimestamps)
        #print 'nofile', nofile
        #s=0
        data_1=np.zeros((2300,2300))
        data_2=np.zeros((2300,2300))
        freqlist1=[]
        freqlist2=[]
        #print 'this is =my files :', myfiles
        #print 'len of list_sametimestamps', len(list_sametimestamps)
        for i in myfiles:
            print 'this is first file', i
        #for i in range(len(myfiles)):

            set1.extend([i for s in set1_subbands if s in i])
            #set1.append([i for s in set1_subbands if s in i])
            #set1.append([s for s in i if s in set1_subbands])
            set2.extend([i for s in set2_subbands if s in i])
            #set2.append([i for s in set2_subbands if s in i])
            number_set1=len([i for i in set1 if i[:]])
            number_set2=len([i for i in set2 if i[:]])
        try:
            
            
            data1_template= fits.getdata(set1[0],
                                      header=False)[0,0,:,:] -fits.getdata(set1[0],
                                      header=False)[0,0,:,:] 
            data2_template= fits.getdata(set2[0],
                                      header=False)[0,0,:,:] -fits.getdata(set2[0],
                                      header=False)[0,0,:,:] 
        #except Exception:
        #    pass
      
            if len([i for i in set1 if i[:]])>4:
                for j in range(len(set1)):
                #if s < nofile/2:
                #while s < nofile:
                    #print s,'inside first'
                    print 'this is first file of set1', j, set1[j]
                    data1_template +=fits.getdata(set1[j],
                                      header=False)[0,0,:,:]
                    #data_1 += fits.getdata(set1[j],
                    #                          header=False)[0,0,:,:]
                    #data_1 += fits.getdata(myfiles[i],
                    #                          header=False)[0,0,:,:]
                    freqlist1.append(float(set1[j].split('-')[-2][1:6])) 

                    #data_1 = data_1/(nofile/2)
                data1_template = data1_template/(number_set1)
                #data_1 = data_1/(number_set1)

                    #if nofile % 2 == 0:
                    #    data_1 = data_1/((nofile/2))

                    #if nofile % 2 == 1:
                     #   data_1 = data_1/(math.floor(nofile/2+1))

                    #freqlist1.append(float(set1[j].split('-')[-2][1:6]))    
                    #freqlist1.append(float(myfiles[i].split('-')[-2][1:6]))

            if len([i for i in set2 if i[:]])>5:


                for k in range(len(set2)):
                    print 'this is first file of set1', k, set2[k]
                #if s >= nofile/2:


                    #print 'second if', s
                    data2_template +=fits.getdata(set2[k],
                                      header=False)[0,0,:,:]
                    #data_2 += fits.getdata(set2[k],
                    #                          header=False)[0,0,:,:]
                    #data_2 += fits.getdata(myfiles[i],
                   #                           header=False)[0,0,:,:]
                    #data_2 = data_2/(nofile/2)
                    freqlist2.append(float(set2[k].split('-')[-2][1:6]))
                data2_template =data2_template /(number_set2)
                #data_2 = data_2/(number_set2)

                    #if nofile % 2 == 0:
                    #    data_2 = data_2/((nofile/2))

                    #if nofile % 2 == 1:
                    #    data_2 = data_2/(math.floor(nofile/2))
                    #freqlist2.append(float(set2[k].split('-')[-2][1:6]))
                    #print k, 'SEEETTTTTTTT2'
                    #print i, "iiiiiiii"
                    #freqlist2.append(float(myfiles[i].split('-')[-2][1:6]))
                    #s += 1
                    #print 'end if', s
            #print 'this is i:',i
            #try:
            #filenum1=int(math.floor(len(set1)/2.+1))
            #filenum2=int(math.floor(len(set2)/2.+1))
            #filenum1=int(math.floor(nofile/2.-1))
            #filenum2=int(math.floor(nofile/2.+1))

            #DIR=set1[0].split('/')[6]
            #DIR=myfiles[filenum1].split('/')[6]
            #DIR=myfiles[int(math.floor(nofile/2.-1))].split('/')[6]
            #DIR=myfiles[math.floor(nofile/2.-1)].split('/')[6]

            #valuefreqlst1=round(np.mean(freqlist1)*0.195,3)
            #valuefreqlst2=round(np.mean(freqlist2)*0.195,3)

            #valuefreqlst1=np.mean(freqlist1)*0.195
            #valuefreqlst2=np.mean(freqlist2)*0.195


            #fitsimg1 = fits.open(set1[0])[0]
            #fitsimg2 = fits.open(set2[0])[0]
            #fitsimg1 = fits.open(myfiles[filenum1])[0]
            #fitsimg2 = fits.open(myfiles[filenum2])[0]
            #fitsimg1 = fits.open(myfiles[math.floor(nofile/2.-1)])[0]
            #fitsimg2 = fits.open(myfiles[math.floor(nofile/2.+1)])[0]
            #fitsimg1.data=data_1
            #fitsimg2.data=data_2
            #try:
            DIR=set1[0].split('/')[6]
            #filenum1=int(math.floor(len(set1)/2.+1))

            valuefreqlst1=np.mean(freqlist1)*0.195
            fitsimg1 = fits.open(set1[0])[0]
            fits.setval(set1[0],'CRVAL3', value=valuefreqlst1)
            fits.setval(set1[0],'RESTFRQ', value=valuefreqlst1)
            fits.setval(set1[0],'RESTFREQ', value=valuefreqlst1)
            fitsimg1.data=data1_template
            #fitsimg1.data=data_1
            datetime1=set1[0].split('/')[-1][:19]
            notgetSB1=round(valuefreqlst1,0)
            filename1= '%s.fits' % (datetime1 + "-S" + str(notgetSB1))
            #except Exception:
            #    pass

            #fits.setval(myfiles[filenum1],'CRVAL3', value=valuefreqlst1)
            #fits.setval(myfiles[filenum1],'RESTFRQ', value=valuefreqlst1)
            #fits.setval(myfiles[filenum1],'RESTFREQ', value=valuefreqlst1)


            #fits.setval(myfiles[math.floor(nofile/2.-1)],'CRVAL3', value=valuefreqlst1)
            #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFRQ', value=valuefreqlst1)
            #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFREQ', value=valuefreqlst1)

            #print 'firs valufreqlist',valuefreqlst1
            #print valuefreqlst2, freqlist2
            #try:
            #filenum2=int(math.floor(len(set2)/2.+1))
            valuefreqlst2=np.mean(freqlist2)*0.195  
            fitsimg2 = fits.open(set2[0])[0]
            fits.setval(set2[0],'CRVAL3', value=valuefreqlst2)
            fits.setval(set2[0],'RESTFRQ', value=valuefreqlst2)
            fits.setval(set2[0],'RESTFREQ', value=valuefreqlst2)
            fitsimg2.data=data2_template
            #fitsimg2.data=data_2
            datetime2=set2[0].split('/')[-1][:19]
            notgetSB2=round(valuefreqlst2,0)

            filename2= '%s.fits' % (datetime2 + "-S" + str(notgetSB2))
            #except Exception:
            #    pass
            #fits.setval(myfiles[filenum2],'CRVAL3', value=valuefreqlst2)
            #fits.setval(myfiles[filenum2],'RESTFRQ', value=valuefreqlst2)
            #fits.setval(myfiles[filenum2],'RESTFREQ', value=valuefreqlst2)


            #fits.setval(myfiles[math.floor(nofile/2.+1)],'CRVAL3', value=valuefreqlst2)
            #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFRQ', value=valuefreqlst2)
            #fits.setval(myfiles[math.floor(nofile/2.-1)],'RESTFREQ', value=valuefreqlst2)

            #myfiles[math.floor(nofile/2-1)].split('/')[-1].split('-')[3]
            #myfiles[math.floor(nofile/2+1)].split('/')[-1].split('-')[3]
            #myfiles[index].split('/')[-1].split('-')[3]

            #datetime1=set1[0].split('/')[-1][:19]
            #datetime2=set2[0].split('/')[-1][:19]
            #except:


            #datetime1=myfiles[filenum1].split('/')[-1][:19]
            #datetime2=myfiles[filenum2].split('/')[-1][:19]
            #datetime1=myfiles[math.floor(nofile/2.-1)].split('/')[-1][:19]
            #datetime2=myfiles[math.floor(nofile/2.+1)].split('/')[-1][:19]

            #getSB1=int(np.floor((1024./200.)*valuefreqlst1))
            #getSB2=int(np.floor((1024./200.)*valuefreqlst2))

            #notgetSB1=round(valuefreqlst1,0)
            #notgetSB2=round(valuefreqlst2,0)


            #filename1= '%s.fits' % (datetime1 + "-S" + str(notgetSB1))
            #filename2= '%s.fits' % (datetime2 + "-S" + str(notgetSB2))

            #print 'wrting path'
            
            #_path='/zfs/helios/filer1/idayan/CALed/AVERAGED2/%s' % (DIR)
            #_path='/zfs/helios/filer1/idayan/CALed/AVERAGED/%s' % (DIR)
        except Exception as e:
            print 'first exception', e
        pass 
            
            
        try:

            _path='/zfs/helios/filer1/idayan/UCALED/AVERAGED-UCAL/%s' % (DIR)
            if os.path.exists(_path):
                print 'writing images' 
                fitsimg1.writeto('/zfs/helios/filer1/idayan/UCALED/AVERAGED-UCAL/%s' % (DIR)+ '/' + filename1, overwrite=True)
                fitsimg2.writeto('/zfs/helios/filer1/idayan/UCALED/AVERAGED-UCAL/%s' % (DIR)+ '/' + filename2, overwrite=True)
            if not os.path.exists(_path):
                os.makedirs(_path)
                print 'writing images, dir created' 
                fitsimg1.writeto('/zfs/helios/filer1/idayan/UCALED/AVERAGED-UCAL/%s' % (DIR)+ '/' + filename1, overwrite=True)
                fitsimg2.writeto('/zfs/helios/filer1/idayan/UCALED/AVERAGED-UCAL/%s' % (DIR) +'/' + filename2, overwrite=True)
        except Exception as p:
            print 'second exception', p
        pass
        #except Exception:
        #    pass 
        

    return

if __name__== "__main__":
  
  Parallel(n_jobs=16,backend="multiprocessing", verbose=10)(delayed(img_averager)(i) for i in matching)
