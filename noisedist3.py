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

from scipy.optimize import curve_fit

from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as SBPL


def calqual(img):
    ref_cat = pd.read_csv("~/AARTFAAC_catalogue.csv")
    #ALL_STD=[]
    print "prinnt imglist"
    print img
    flux_compare = []
    f_comp=[]
    flux_correct = []

    fitsimg=fits.open(img)

    configuration = {
            "back_size_x": 64,
            "back_size_y": 64,
            "margin": 0,
            "radius": 0}

    img_HDU = fits.HDUList(fitsimg)
    imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                               plane=0),
                                                 **configuration)

    sr = imagedata.extract(det=5, anl=3,
                           labelled_data=None, labels=[],
                           force_beam=True)

    # Reference catalogue compare
    slope_cor, intercept_cor, ref_match, image_match, index_match, DISTs, STD_, DSTs2= compare_flux(sr,
                                           ref_cat["ra"],
                                           ref_cat["decl"],
                                           ref_cat["f_int"],
                                           ref_cat["f_int_err"],img)
    
    #P0=ZZ
    #P1=[1150.,1150.]
    #Distance  = np.sqrt((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)
    
    #ref_flux=np.array(ref_match)
    #sr_corr_flux=(np.array(image_match) - intercept_cor)/slope_cor
    #print np.array(ref_match)
    #print "ref_flux"
    #print ref_flux
    #print sr_corr_flux
    #std_=np.std(ref_flux/sr_corr_flux)
    f_comp.append([np.array(ref_match),np.array(image_match),DSTs2])
    
    #flux_compare.append([np.array(ref_match),
     #                        (np.array(image_match) - intercept_cor)/slope_cor,
     #                       np.array(np.ravel(index_match)), np.array(DISTs), STD_,DSTs2])

    #flux_compare.append([np.array(ref_match),
    #                         (np.array(image_match) - intercept_cor)/slope_cor,
    #                        np.array(np.ravel(index_match))])

    #flux_correct.append([i, slope_cor, intercept_cor])
    testt = pd.DataFrame([])

    for i in range(len(f_comp)):
        
        if len(testt) == 0:
            
            testt = pd.DataFrame({"reference":f_comp[i][0],
                                 "image":f_comp[i][1],
                                "DIST2":f_comp[i][2]})

        else:
            
            testt = pd.concat([testt, pd.DataFrame({"reference":f_comp[i][0],
                                                    "image":f_comp[i][1],"DIST2":f_comp[i][2]})])
    
    return testt



def distSquared(p0, p1):
    '''
    Calculate the distance between point p0, [x,y], and a list of points p1, [[x0..xn],[y0..yn]]. 
    '''
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 0.5:
        return np.where(distance == np.min(distance))[0]
    else:
        return None

def pol2cart(rho, phi):
    """
    Polar to Cartesian coordinate conversion, for distance measure around celestial pole.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def compare_flux(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs, I):
    '''
    Compares the two catalogues, matching sources, and outputs the results of linear fit to the fluxes. 
    '''
    x = []
    y = []

    w = []
    sr_indexes = []
    cat_indexes = []
    distances_ = []
    STDev=[]
    distances_2 = []


    for i in range(len(sr)):

        sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                np.deg2rad(sr[i].ra.value))

        cat_x, cat_y = pol2cart(np.abs(90-catalog_decs),
                np.deg2rad(catalog_ras))

        index = distSquared((sr_x,sr_y),
                   np.array([cat_x, cat_y]))
       

        if type(index) == np.ndarray:
            print "INDEXXX"
            print index
            print "catalog_fluxs[index]"
            print catalog_fluxs[index]
            print '**********'
            flux = catalog_fluxs[index]
            flux_err = catalog_flux_errs[index]

            cat_indexes.append(index)
            sr_indexes.append(i)
            y.append(float(sr[i].flux.value))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
            #calculate each foundeed sources location to the center
            x_=sr[i].ra.value
            y_=sr[i].dec.value
            #x_=fits.getheader(I)['CRVAL1']
            #y_=fits.getheader(I)['CRVAL2']
            
            w_ = wcs.WCS(fits.getheader(I))
            spot=w_.wcs_world2pix(x_,y_,0,0,0)
            P1=[1150.,1150.]
            distances_2.append(np.sqrt((spot[0] - P1[0])**2 + (spot[1] - P1[1])**2))
            
            #spot
            P0=[sr_x, sr_y]
            P1=[1150.,1150.]
            distances_.append(np.sqrt((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2))
            #calculate std 
            STDev.append(float(sr[i].flux.value)/float(flux))
            
    
        else:
            continue

    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]
    print "DITANCE"
    print distances_

    return fit[0], fit[1], x, y, cat_indexes, distances_, STDev,distances_2 


#global calqual



with open("/home/idayan/ALL-TXT/avrgucal202009290730.txt",'r') as f: # 12k
    lines=f.read().splitlines()
images = sorted(lines)

#images = images[4220:8440]
#images = images[6330:8440]
images = images[8440:10550]

srimages=pd.Series(images)
srimages_57=srimages[srimages.str.contains("-S57.0") | srimages.str.contains("-S58.0") | srimages.str.contains("-S56.0")]
srimages_62=srimages[srimages.str.contains("-S62.0") | srimages.str.contains("-S63.0") | srimages.str.contains("-S60.0")| srimages.str.contains("-S61.0")]


if __name__ == "__main__":
    
    print "creating df"
    A9_57=pd.DataFrame()
    print "created a9 data frame"
    A9_57=A9_57.append(Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(calqual)(i) for i in srimages_57[:600]))
    print "multiprocessing"
    A9_57.to_pickle("/home/idayan/noisedist_testA9_57_topickle.pkl")

#A9_62=pd.DataFrame()
#A9_62=A9_62.append(Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(calqual)(i) for i in srimages_62[:600]))
#A9_62.to_pickle("/home/idayan/noisedist_testA9_62_topickle.pkl")



