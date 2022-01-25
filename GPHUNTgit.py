import os
import sys
import csv 
import argparse
import glob
import time 

from datetime import datetime
import numpy as np
import pandas as pd

#from sourcefinder.accessors import open as open_accessor
#from sourcefinder.accessors import sourcefinder_image_from_accessor

from joblib import Parallel, delayed

from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.time import Time

from tkp.accessors import sourcefinder_image_from_accessor
from tkp.accessors import open as open_accessor

from astropy.coordinates import SkyCoord

#loca=SkyCoord.from_name("PSR B0950+08")

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
            y.append(float(sr[i].flux))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
        else:
            continue
            
    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]

    return fit[0], fit[1]

def Check_location(fits_file):
    #target_ra = 293.732
    #target_dec = 21.896
    target_ra = 148.289
    target_dec = 7.927
    
    fitsimg = fits.open(fits_file)[0]
    
    lofarfrequencyOffset = 0.0
    lofarBW = 195312.5

    ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")

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

        # Reference catalogue compare
        slope_cor, intercept_cor = compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])
        
        # look for source detection
        for i in range(len(sr)):
            sr_x, sr_y = pol2cart(np.abs(90.-sr[i].dec.value),
                    np.deg2rad(sr[i].ra.value))

            cat_x, cat_y = pol2cart(np.abs(90.-target_dec),
                    np.deg2rad(target_ra))

            distance  = np.sqrt((sr_x - cat_x)**2 + (sr_y - cat_y)**2)
            if  distance < 1.0:
                
                fields=[(sr[i].flux.value-intercept_cor)/slope_cor,
                        (sr[i].flux.error)/slope_cor,
                        sr[i].ra.value, sr[i].ra.error,
                        sr[i].dec.value, sr[i].dec.error, 
                        frq, datetime.strftime(t.datetime, format="%Y-%m-%dT%H:%M:%S"), fits_file, None]
        
            #replace with pandas csv write 
                with open(r'/home/idayan/_GPhunt.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)




