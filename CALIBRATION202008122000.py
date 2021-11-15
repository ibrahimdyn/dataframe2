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



def distSquared(p0, p1):
    '''
    Calculate the distance between point p0, [x,y], and a list of points p1, [[x0..xn],[y0..yn]]. 
    '''
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 1.0:
        return np.where(distance == np.min(distance))[0]
    else:
        return None
    

def pol2cart(long,lat):
     """
     Utility function to convert longitude,latitude on a unit sphere to
     cartesian co-ordinates.
     """
 
     x=np.cos(lat)*np.cos(long)
     y=np.cos(lat)*np.sin(long)
     z=np.sin(lat)
     return np.array([x,y])

def compare_flux(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs):
    '''
    Compares the two catalogues, matching sources, and outputs the results of linear fit to the fluxes. 
    '''
    x = []
    y = []

    w = []
    sr_indexes = []
    cat_indexes = []


    for i in range(len(sr)):

        sr_x, sr_y = pol2cart(sr[i].ra.value,
                np.deg2rad(sr[i].dec.value))

        cat_x, cat_y = pol2cart(catalog_ras,
                np.deg2rad(catalog_decs))

        index = distSquared((sr_x,sr_y),
                   np.array([cat_x, cat_y]))

        if type(index) == np.ndarray:
            flux = catalog_fluxs[index]
            flux_err = catalog_flux_errs[index]

            cat_indexes.append(index)
            sr_indexes.append(i)
            y.append(float(sr[i].flux.value))
            x.append(float(flux))
            #w.append(float(sr[i].f_int_err))
        else:
            continue
    
    fit = np.polyfit(x,y,1)
    print(x)
    return fit[0], fit[1]
  
  
def calibrate(images):
    print "running process"
    fitsimg = fits.open(images)[0]
    #pathofimage=images
    
    print os.path.basename(images)
    
    if np.nanstd(fitsimg.data[0,0,:,:]) < 100 :
        
        #look for sources
        configuration = {
            "back_size_x": 64,
            "back_size_y": 64,
            "margin": 0,
            "radius": 0}

        img_HDU = fits.HDUList(fitsimg)
        imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                                   plane=0),
                                                     **configuration)
        print "before sr"

        sr = imagedata.extract(det=5, anl=3,
                               labelled_data=None, labels=[],
                               force_beam=True)
        print "after sr"


        if fitsimg.header['CRVAL3']<44*10**6:
            #ref_cat_8C=pd.read_csv('~/asu.tsv', sep=';',error_bad_lines=False, comment='#')
            print "inside first if"
        
            ref_cat_8C = pd.read_csv('~/asu.tsv', sep=';',error_bad_lines=False, comment='#')
            #zz = SkyCoord((ref_cat_8C['RAB1950']),(ref_cat_8C['DEB1950']), unit=(u.hourangle, u.deg))
            slope_cor, intercept_cor = compare_flux(sr,
                                       SkyCoord((ref_cat_8C['RAB1950']),(ref_cat_8C['DEB1950']), unit=(u.hourangle, u.deg)).ra.deg,
                                       SkyCoord((ref_cat_8C['RAB1950']),(ref_cat_8C['DEB1950']), unit=(u.hourangle, u.deg)).dec.deg,
                                       ref_cat_8C["Sint"],
                                       ref_cat_8C["Sint"]/ref_cat_8C["S/N"])
            print "still inside first if, check slope_cor:" , slope_cor
        else:
            ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")
            slope_cor, intercept_cor = compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])
            print "ssecond if, check slope_cor:" , slope_cor
            
        fields=[slope_cor, intercept_cor, len(sr), os.path.basename(images)]
        with open(r'/home/idayan/FIT_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            
        fitsimg.data[0,0,:,:] = (fitsimg.data[0,0,:,:]-intercept_cor)/slope_cor
        #fitsimg.writeto(cfg.outdir+filename,overwrite=True)
        fitsimg.writeto("/home/idayan/CALIBRATEDIMAGES/202008122000/"+os.path.basename(images),overwrite=True)
        print("didwritethe image")
        
        #os.remove(cfg.indir+cfg.fitsfile)    
#vv 
fits_list=glob.glob(obs_dir+"202008122000/"+"*_all*"+"/*SB*/"+"imgs/"+"*")
for i in fits_list[0:200]:
    calibrate(i)
