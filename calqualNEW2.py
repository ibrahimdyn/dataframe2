import os
import sys
import csv

#from datetime import datetime
import datetime

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

#from astropy.visualization import simple_norm
#import matplotlib.pyplot as plt
#from photutils.aperture import CircularAperture, CircularAnnulus
#from photutils.datasets import make_100gaussians_image

from astropy.stats import sigma_clipped_stats
import pickle

import tkp.db.alchemy
from pandas import DataFrame
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import DatetimeTickFormatter

logging.basicConfig(level=logging.INFO)
output_notebook()

query_loglevel = logging.WARNING 

from scipy.optimize import curve_fit

from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as SBPL

import h5py
from scipy import interpolate

from multiprocessing.pool import ThreadPool as Pool

import time


start=time.time()

IMG = sys.argv[1]

def get_beam(freq):
    beams = np.array([30,35,40,50,55,60,65,70,75,80,85,90])
    freq_to_use = str(beams[np.argsort(np.abs(freq-beams))[0]])

    beam_file = "/home/idayan/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_{}MHz.hdf5".format(freq_to_use)
    #beam_file = "/home/mkuiack1/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_{}MHz.hdf5".format(freq_to_use)
    orig =  np.array(h5py.File(beam_file, mode="r").get('lmbeamintensity_norm'))

    # Make its coordinates; x is horizontal.
    x = np.linspace(0, 2300, orig.shape[1])
    y = np.linspace(0, 2300, orig.shape[0])

    # Make the interpolator function.
    f = interpolate.interp2d(x, y, orig, kind='linear')

    # Construct the new coordinate arrays.
    x_new = np.arange(0, 2300)
    y_new = np.arange(0, 2300)

    # Do the interpolation.
    return f(x_new, y_new)




def pol2cart(rho, phi):
    """
    Polar to Cartesian coordinate conversion, for distance measure around celestial pole.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def distSquared3(p0, p1):
    '''
    Calculate the distance between point p0, [x,y], and a list of points p1, [[x0..xn],[y0..yn]]. 
    '''
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 0.5:
        print distance
        print "POSIIONNNNNN"
        #print np.where(distance == np.min(distance))
        print np.where(distance == np.min(distance))[0][0]
        return np.where(distance == np.min(distance))[0]
    else:
        return None
    
def compare_flux3(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs, I):
    '''
    Compares the two catalogues, matching sources, and outputs the results of linear fit to the fluxes. 
    '''
    x = []
    y = []

    w = []
    sr_indexes = []
    cat_indexes = []
    distances_2 = []
    
    sr_err = []
    cat_err =[]
    
    for i in range(len(sr)):
        
        
        print sr[i].flux.value
        if (sr[i].flux.value>10.) & (sr[i].flux.value<1600):
        #if (sr[i].flux.value>0):
            print "SR FLUX VALUE"
            print sr[i].flux.value


            sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                    np.deg2rad(sr[i].ra.value))

            cat_x, cat_y = pol2cart(np.abs(90-catalog_decs),
                    np.deg2rad(catalog_ras))

            index = distSquared3((sr_x,sr_y),
                       np.array([cat_x, cat_y]))
            print "INDEXXX"
            print index
            #print "CATFLUX"
            #print catalog_fluxs[index]
            #print "sr"

            if type(index) == np.ndarray:
                flux = catalog_fluxs[index]
                flux_err = catalog_flux_errs[index]
                print "FFFLLLLLLLLUUUUXXXXXXXX_err"
                print flux_err
                cat_err.append(catalog_flux_errs[index[0]])
                #cat_err.append(catalog_flux_errs[index][0])

                cat_indexes.append(index)
                sr_indexes.append(i)
                y.append(float(sr[i].flux.value))
                x.append(float(flux))
                w.append(float(sr[i].flux.error))
                sr_err.append(float(sr[i].flux.error))

                x_=sr[i].ra.value
                y_=sr[i].dec.value
                w_ = wcs.WCS(fits.getheader(I))
                spot=w_.wcs_world2pix(x_,y_,0,0,0)
                P0=[sr_x, sr_y]
                P1=[1151.,1151.]
                distances_2.append(np.sqrt((spot[0] - P1[0])**2 + (spot[1] - P1[1])**2))
            else:
                continue
    #else:
       # continue

    if len(x) > 2:
        print "XXX"
        print x
        print "YYY"
        print y
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]
    print x
    print y

    return fit[0], fit[1], x, y, cat_indexes, distances_2, sr_err, cat_err 
  
  
AA=pd.DataFrame([])
BB=[]
#for i,img in enumerate(imglst):
#for i,img in enumerate(imgs[200:202]):
#for i,img in enumerate(caledimgs0704[7000:7003]):
#for i,img in enumerate(imgs1814caledwbc[50:53]):
#for i,img in enumerate(imgs0605[50:53]):
#for i,img in enumerate(imgs1003[60:70]):

### YOU SHOULD DEFINE A FUNCTION 
#def imgcalqual():
#for img in IMG: #!!! for i,img in enumerate(imgs1003[60:70]):
#for img in IMG: #!!! for i,img in enumerate(imgs1003[60:70]):
CC=[]

#!!! print i
print IMG
print "IMAGE PRINTED"

#for i,img in enumerate(img): 
#for i,img in enumerate(imglst[10:20]):  

#for i,img in enumerate(imgss[10:20]):  

#fitsimg=fits.open(imglst[7])
#fitsimg=fits.open(img)
def qualcalculator(pIMG):
  
  fitsimg=fits.open(pIMG)[0]

  #print fitsimg.data[:,:], "BEFORE"


  #bg_data, bg_f =fits.getdata(img[0], header=True)
  bg_data, bg_f =fits.getdata(pIMG, header=True)
  print fitsimg.data[0,0,:,:]
  beam_model = get_beam(bg_f["CRVAL3"]/1e6)

  fitsimg.data[0,0,:,:]=fitsimg.data[0,0,:,:]*(np.max(beam_model)/beam_model)
  #print fitsimg.data[0,0,:,:]


  ref_cat = pd.read_csv("~/AARTFAAC_catalogue.csv")
  flux_compare=[]
  configuration = {
          "back_size_x": 64,
          "back_size_y": 64,
          "margin": 0,
          "radius": 0}

  img_HDU = fits.HDUList(fitsimg)
  imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                             plane=0),
                                               **configuration)
  #print fitsimg.data[:,:], "AFTER"
  sr = imagedata.extract(det=3, anl=3,
                             labelled_data=None, labels=[],
                             force_beam=True)

      # Reference catalogue compare
  slope_cor, intercept_cor, ref_match, image_match, index_match, DST2,SR_err,CAT_err = compare_flux3(sr,
                                             ref_cat["ra"],
                                             ref_cat["decl"],
                                             ref_cat["f_int"],
                                             ref_cat["f_int_err"], IMG)
  flux_compare.append([np.array(ref_match),
                        #  (np.array(image_match))/slope_cor,
                          (np.array(image_match) - intercept_cor)/slope_cor,
                   #(np.array(image_match)* slope_cor ) + intercept_cor,
                          np.array(np.ravel(index_match)),np.array(image_match), 
                   DST2, SR_err, CAT_err ])

  #flux_compare.append([ np.array(ref_match),
                            #  (np.array(image_match))/slope_cor,
  #                            (np.array(image_match) - intercept_cor)/slope_cor,
                       #(np.array(image_match)* slope_cor ) + intercept_cor,
  #                            np.array(np.ravel(index_match)),np.array(image_match), 
   #                    DST2, SR_err, CAT_err ])

      #flux_correct.append([i, slope_cor, intercept_cor])

  test = pd.DataFrame([])

  for i in range(len(flux_compare)):

      if len(test) == 0:
          test = pd.DataFrame({"reference":flux_compare[i][0],
                               "image":flux_compare[i][1],
                              "IMnorm":flux_compare[i][3],
                              "DST":flux_compare[i][4],
                              "SR-ERR":flux_compare[i][5],
                              "CAT-ERR":flux_compare[i][6]},
                              index=flux_compare[i][2])

      else:
          test = pd.concat([test,
                 pd.DataFrame({"reference":flux_compare[i][0],
                               "image":flux_compare[i][1],
                              "IMnorm":flux_compare[i][3],
                              "DST":flux_compare[i][4],
                              "SR-ERR":flux_compare[i][5],
                              "CAT-ERR":flux_compare[i][6]},
                              index=flux_compare[i][2])])

  STD=np.std(test.image/test.reference)
  
  return STD
#with open('/home/idayan/calqualNEWWW.pkl', 'wb+') as f00:
    
    #with open('/home/idayan/GPsearch07DF1-5.pkl', 'wb') as ff:
#    pickle.dump([STD], f00)
#with open('/home/idayan/calqualNEWWWWB.pkl', 'wb') as f00W:
with open('/home/idayan/calqualNEWWWWB0604.pkl', 'wb') as f00W:
    
    #with open('/home/idayan/GPsearch07DF1-5.pkl', 'wb') as ff:
    pickle.dump([qualcalculator(IMG)], f00W)

#AA=pd.concat([AA,test])
BB.append(qualcalculator(IMG))
CC.append(qualcalculator(IMG))
#with open('/home/idayan/calqualNEW-CCB.pkl', 'rb+') as f0:  
#with open('/home/idayan/GPsearch07DF1-5.pkl', 'wb') as ff:
#    pickle.dump(CC, f0)

    #STD=np.std(test.image/test.reference)
    #ALL_STD.append(STD)
    #return STD, test.index

#with open('/home/idayan/calqualNEW.pkl', 'rb') as f1: # rb is for only reading
#with open('/home/idayan/GPsearch07DF1-5.pkl', 'wb') as ff:
#    pickle.dump([BB], f1)

#with open('/home/idayan/calqualNEWW.pkl', 'a+') as f2:
with open('/home/idayan/calqualNEWW0604.pkl', 'a+') as f2:
#with open('/home/idayan/GPsearch07DF1-5.pkl', 'wb') as ff:
    pickle.dump(BB, f2)

#with open(r'/home/idayan/calqualNEWcsvB.csv', 'a+') as f22:
with open(r'/home/idayan/calqualNEWcsvB0604.csv', 'a+') as f22:
    writer = csv.writer(f22)
    writer.writerow([qualcalculator(IMG)])

