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

from astropy.coordinates import SkyCoord, match_coordinates_sky,  AltAz, EarthLocation

from photutils.datasets import make_100gaussians_image

from astropy.visualization import simple_norm


from photutils.datasets import make_100gaussians_image
from photutils.aperture import CircularAperture, CircularAnnulus

from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.datasets import make_100gaussians_image

from astropy.stats import sigma_clipped_stats



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

#fits_list = sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202005051300/2*.fits"))
#fits_list = sorted(glob.glob("/zfs/helios/filer1/idayan/CALed/202006040630/2*.fits"))

with open("/home/idayan/imgsin60.txt",'r') as f:
    lines=f.read().splitlines()
imagess=lines


lofarfrequencyOffset = 0.0
lofarBW = 195312.5

ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")


count = 0

dff=pd.DataFrame()
column_names=['MEDofSOURCE','BACKGROUND','DATE-OBS']
listofres=[]

start=time.time()
print "startingforloop:"
for fits_file in imagess:
#for fits_file in fits_list[0:20]:
#     start_t = time.time()
    print "first image:", fits_file
    print "count is:", count
    if count % len(imagess) == 0:
        print count, "/", len(imagess)
        
    fitsimg = fits.open(fits_file)[0]
    
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
            "radius": 770}

        img_HDU = fits.HDUList(fitsimg)
        imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                                   plane=0),
                                                     **configuration)

        sr = imagedata.extract(det=5, anl=3,
                               labelled_data=None, labels=[],
                               force_beam=True)

        # Reference catalogue compare
#         slope_cor, intercept_cor = compare_flux(sr,
#                                        ref_cat["ra"],
#                                        ref_cat["decl"],
#                                        ref_cat["f_int"],
#                                        ref_cat["f_int_err"])
        # look for source detection 
        for i in range(len(sr)):
            sr_x, sr_y = pol2cart(np.abs(90-sr[i].dec.value),
                    np.deg2rad(sr[i].ra.value))

            cat_x, cat_y = pol2cart(np.abs(90-7.66),
                    np.deg2rad(148.56))

            distance  = np.sqrt((sr_x - cat_x)**2 + (sr_y - cat_y)**2)
            if  distance < 1.0:
                print count
                print "FOUND"
                print fits_file
                print sr[i]
                
                fitsimgdata=fits.getdata(fits_file)[0,0,:,:]
                hdu_1 = fits.getheader(fits_file)

                wks = wcs.WCS(hdu_1)
                wkks = wks.dropaxis(-1).dropaxis(-1)

                positionsky= [(148.56, 7.66)]
                pixposs=wkks.all_world2pix(positionsky,1)
                #positionspix=pixposs[0]
                aperture = CircularAperture(pixposs, r=12.78)
                annulus_aperture = CircularAnnulus(pixposs, r_in=24, r_out=36)


                annulus_masks = annulus_aperture.to_mask(method='center')
                annulus_data = annulus_masks[0].multiply(fitsimgdata)
                mask = annulus_masks[0].data
                annulus_data_1d = annulus_data[mask > 0]
                #annulus_data_1d.shape
                _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
                background = median_sigclip * aperture.area()
                print(background) 



                annulus_aperture_Source = CircularAnnulus(pixposs, r_in=0.001, r_out=12.78)
                annulus_masks_Source =annulus_aperture_Source.to_mask(method='center')
                annulus_data_Source=annulus_masks_Source[0].multiply(fitsimgdata)
                mask_Source = annulus_masks_Source[0].data
                annulus_data_1d_Source = annulus_data_Source[mask_Source > 0]
                #annulus_data_1d_Source.shape
                Medofsourcepoint=np.median(annulus_data_1d_Source)
                print(np.median(annulus_data_1d_Source))

                Dateobs=hdu_1["DATE-OBS"]
                #Date-obs=hdu_1["DATE-OBS"]
                print(Dateobs)
                #templist=[Medofsourcepoint,background,Dateobs]
                templist=[Medofsourcepoint,median_sigclip,Dateobs]

                listofres.append(templist)
                fields=['fitsimg:',fits_file,"sr[i]:",sr[i],'sig,backg,date:',templist]
                with open(r'/home/idayan/GPsearch07.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
            #if distance > 2.0:
                #print fitsimg
                #print sr[i]
                #print count

    count += 1
    
DFF=pd.DataFrame(listofres[0:],columns=column_names)
end=time.time()
print "duration:", end-start
