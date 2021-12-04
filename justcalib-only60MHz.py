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
import pyds9 as pyd
from tkp.accessors import sourcefinder_image_from_accessor
from tkp.accessors import open as open_accessor


from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

import csv

import tkp.sourcefinder.image
import matplotlib.pyplot as plt
import math
#from datetime import datetime





def distSquared(p0, p1):
    '''
    Calculate the distance between point p0, [x,y], and a list of points p1, [[x0..xn],[y0..yn]]. 
    '''
    distance  = np.sqrt((p0[0] - p1[0,:])**2 + (p0[1] - p1[1,:])**2)
    if np.min(distance) < 1.0:
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
    

def polar2cart(long,lat):
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

        #sr_x, sr_y = pol2cart(sr[i].ra.value,
        #        np.deg2rad(sr[i].dec.value))

        #cat_x, cat_y = pol2cart(catalog_ras,
        #        np.deg2rad(catalog_decs))
        sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                np.deg2rad(sr[i].ra.value))

        cat_x, cat_y = pol2cart(np.abs(90-catalog_decs),
                np.deg2rad(catalog_ras))

        index = distSquared((sr_x,sr_y),
                   np.array([cat_x, cat_y]))

        if type(index) == np.ndarray:
            flux = catalog_fluxs[index]
            flux_err = catalog_flux_errs[index]

            cat_indexes.append(index)
            sr_indexes.append(i)
            y.append(float(sr[i].flux.value))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
        else:
            continue
            
    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit,cov = np.polyfit(x,y,1,w=1./w,cov=True)
    else:
        fit = [1e10,1e10,1e10,1e10,0]
        cov = np.array([[1e12, 1e12], [1e12, 1e12]])

    return fit[0], cov[0,0], fit[1], cov[1,1], len(x)
    #fit = np.polyfit(x,y,1)
    #3print(x)
    #return fit[0], fit[1]

#loca=SkyCoord.from_name("PSR B0950+08")
ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")

def Check_location(fits_file):
    #target_ra = 293.732
    #target_dec = 21.896
    #target_ra= loca.ra.deg
    #target_dec= loca.dec.deg
    print fits_file
    
    fitsimg = fits.open(fits_file)[0]
    
    
    lofarfrequencyOffset = 0.0
    lofarBW = 195312.5

    #ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")

    t = Time(fitsimg.header['DATE-OBS'])
    frq = fitsimg.header['RESTFRQ']
    bw = fitsimg.header['RESTBW']


    
    # Initial quality condition. 
    if np.nanstd(fitsimg.data[0,0,:,:]) < 1000.0:

        # Source find 
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
        print len(sr), "srs:", sr
        
        #ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")
        
        slope_cor,slope_err, intercept_cor, int_err, N_match = compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])
        
        fields=[slope_cor,slope_err, intercept_cor, int_err, N_match, len(sr)]
        with open(r'/home/idayan/CALwith60Mhz/Calimgs/fit_results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        
        if slope_cor < 1e8:
            filename = '%s.fits' % (datetime.fromtimestamp(t.unix).strftime('%Y-%m-%dT%H:%M:%S')+ \
                            "-S"+str(round((frq-lofarfrequencyOffset)/lofarBW,1))+ \
                            "-B"+str(int(np.ceil(bw /lofarBW))))

            fitsimg.data[0,0,:,:] = (fitsimg.data[0,0,:,:]-intercept_cor)/slope_cor
            fitsimg.writeto("/home/idayan/CALwith60Mhz/Calimgs/"+filename,overwrite=True)
        
        
        

        
        
                    

obs_dir = "/zfs/helios/filer0/mkuiack1/"
#fits_list=glob.glob(obs_dir+"202008122000/"+"*_all*"+"/*SB*/"+"imgs/"+"*")
fits_list=sorted(glob.glob(obs_dir+"202008122000/"+"*_all*"+"/*SB*/"+"imgs/"+"*"))

t1 = time.time()

#for i in fits_list:
#    calibrate(i)
    
Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(Check_location)(i) for i in fits_list)
print "processing time:", time.time() -t1
