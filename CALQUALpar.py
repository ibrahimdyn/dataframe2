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

from multiprocessing.pool import ThreadPool as Pool


import time
start=time.time()

IMG= sys.argv[1]

def cmpltcalqual(img):
    ref_cat = pd.read_csv("~/AARTFAAC_catalogue.csv")
    #ALL_STD=[]
    print "prinnt imglist"
    print img
    flux_compare = []
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
    slope_cor, intercept_cor, ref_match, image_match, index_match = compare_flux(sr,
                                           ref_cat["ra"],
                                           ref_cat["decl"],
                                           ref_cat["f_int"],
                                           ref_cat["f_int_err"])

    flux_compare.append([np.array(ref_match),
                             (np.array(image_match) - intercept_cor)/slope_cor,
                            np.array(np.ravel(index_match))])

    #flux_correct.append([i, slope_cor, intercept_cor])

    test = pd.DataFrame([])

    for i in range(len(flux_compare)):

        if len(test) == 0:
            test = pd.DataFrame({"reference":flux_compare[i][0],
                                 "image":flux_compare[i][1]},
                                index=flux_compare[i][2])

        else:
            test = pd.concat([test,
                   pd.DataFrame({"reference":flux_compare[i][0],
                                 "image":flux_compare[i][1]},
                                index=flux_compare[i][2])])

    STD=np.std(test.image/test.reference)
    #ALL_STD.append(STD)
    return STD#stdintegrator(STD)

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
            y.append(float(sr[i].flux.value))
            x.append(float(flux))
            w.append(float(sr[i].flux.error))
        else:
            continue

    if len(x) > 2:
        w = np.array(w,dtype=float)
        fit = np.polyfit(x,y,1,w=1./w)
    else:
        fit = [1e9,1e9]

    return fit[0], fit[1], x, y, cat_indexes


imgs=sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202006040830/2*.fits"))

def stdintegrator(args):
    a=[]
    a.append(args)
    print a
    srs=pd.Series(a)
    print srs
    return srs


print "executig the function"
#cmpltcalqual(IMG)

with open(r'/home/idayan/TESTforcalqual.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(cmpltcalqual(IMG))
#if __name__ == "__main__":
  
#fitslist=imgs
#pool.apply_async(foo.work)
#out=Pool.apply_async(Parallel(n_jobs=5,backend="multiprocessing", verbose=10)(delayed(cmpltcalqual)(i) for i in fitslist) )
#temp=[]
#temp.append(Parallel(n_jobs=5,backend="multiprocessing", verbose=10)(delayed(cmpltcalqual)(i) for i in fitslist) )

#out=Parallel(n_jobs=5,backend="multiprocessing", verbose=10)(delayed(cmpltcalqual)(i) for i in fitslist) 
#finalres=pd.DataFrame(temp)
#finalres.to_pickle("/home/idayan/STD1.pkl")
#finalres.to_csv("/home/idayan/STDlast.csv")
    #[(integrate(i)) for i in fitslist[0:3]]
    #EDF=pd.DataFrame
    #out=Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(noisedist)(i) for i in fitslist)
    #out2=Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(noisedist)(i) for i in fitslist2)
    #finalres=pd.DataFrame(out)
    #finalres2=pd.DataFrame(out2)
    #finalres.to_pickle("/home/idayan/noisegraphdfLAST000Co202005051300.pkl") 
    #finalres.to_pickle("/home/idayan/newnoisegrapDF/AVRimgs10inc57-{}.pkl".format(obs_folder))
    #finalres2.to_pickle("/home/idayan/newnoisegrapDF/AVRimgs10inc62-{}.pkl".format(obs_folder))
    #finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfLAST000Co-{}.pkl".format(obs_folder)) 
    #EDF.append(pd.DataFrame(out))
    #DF.append(pd.Series(out))
    #pd.DataFrame(out)

    #EDF=EDF.append(pd.DataFrame(pd.Series(out)))
    #EDF.append((out))



end=time.time()
print("processing time is:",end-start) 
