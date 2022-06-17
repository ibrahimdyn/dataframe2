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
import pickle




IM=sys.argv[1]
print "executtion started lc-anulus-analyze.py"
print IM
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


lofarfrequencyOffset = 0.0
lofarBW = 195312.5

ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")

count = 0

dff=pd.DataFrame()
column_names=['MEDofSOURCE','MAXofSOURCE','STDEV-BACKGR','DATE-OBS']
headerList=['IMAGE','MEDofSOURCE','MAXofSOURCE','STDEV-BACKGR','DATE-OBS']
listofres=[]

start=time.time()
print "startingforloop:"


#def gphunter(IM):
def lc_annulus(IM):

    print 'first image in the loop:', IM
    #print "count is:", count
   # if count % len(imagess) == 0:
   #     print count, "/", len(imagess)
    # Initial quality condition. 
    fitsimg = fits.open(IM)[0]
    if (np.nanstd(fitsimg.data[0,0,:,:]) < 1000.0) & (np.nanstd(fitsimg.data[0,0,:,:]) > 1.0):
    #if np.nanstd(fitsimg.data[0,0,:,:]) < 1000.0:
        
        
    
        t = Time(fitsimg.header['DATE-OBS'])
        frq = fitsimg.header['RESTFRQ']
        bw = fitsimg.header['RESTBW']

        
        fitsimgdata=fits.getdata(IM)[0,0,:,:]
        hdu_1 = fits.getheader(IM)
        #fitsimgdata=fits.getdata(fits_file)[0,0,:,:]
        #hdu_1 = fits.getheader(fits_file)

        wks = wcs.WCS(hdu_1)
        wkks = wks.dropaxis(-1).dropaxis(-1)

        #positionsky= [(148.56, 7.66)]
        positionsky= [(293.73, 21.90)]
        
        pixposs=wkks.all_world2pix(positionsky,1)
        #positionspix=pixposs[0]
        aperture = CircularAperture(pixposs, r=12.78)
        annulus_aperture = CircularAnnulus(pixposs, r_in=24, r_out=36)


        annulus_masks = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_masks[0].multiply(fitsimgdata)
        mask = annulus_masks[0].data
        annulus_data_1d = annulus_data[mask > 0]
        #annulus_data_1d.shape
        _, median_sigclip, stdevback = sigma_clipped_stats(annulus_data_1d)
        #background = median_sigclip * aperture.area()
        #print(background) 



        annulus_aperture_Source = CircularAnnulus(pixposs, r_in=0.001, r_out=12.78)
        annulus_masks_Source =annulus_aperture_Source.to_mask(method='center')
        annulus_data_Source=annulus_masks_Source[0].multiply(fitsimgdata)
        mask_Source = annulus_masks_Source[0].data
        annulus_data_1d_Source = annulus_data_Source[mask_Source > 0]
        #annulus_data_1d_Source.shape
        #Medofsourcepoint=np.median(annulus_data_1d_Source)
        Medofsrc=np.median(annulus_data_1d_Source)
        print(np.median(annulus_data_1d_Source))
        Maxofsrc=np.max(annulus_data_1d_Source)

        Dateobs=hdu_1["DATE-OBS"]
        #Date-obs=hdu_1["DATE-OBS"]
        #print(Dateobs)
        #templist=[Medofsourcepoint,background,Dateobs]
        #templist=[Medofsourcepoint,stdevback,Dateobs]
        templist=[Medofsrc,Maxofsrc,stdevback,Dateobs]

        listofres.append(templist)
        #fields=['fitsimg:',IM,"sr[i]:",sr[i],'sig,backg,date:',templist]
        #fields=[IM,sr[i],templist]
        fields=[IM,Medofsrc,Maxofsrc,stdevback,Dateobs]
        
        with open(r'/home/idayan/SGRDD2.csv', 'a') as f:
        #with open(r'/home/idayan/LC_ANNULUS_stdup.csv', 'a') as f:
            writer = csv.writer(f)
            #writer.writerow(headerList)
            #writer.DictReader(f, fieldnames=headerList)
            writer.writerow(fields)
            #keys = ['name', 'age', 'job', 'city']
            #reader = csv.DictReader(f, fieldnames=keys)
        with open('/home/idayan/SGRDD2.pkl', 'a') as ff:
        #with open('/home/idayan/LC_ANNULUS_stdup.pkl', 'a') as ff:
            pickle.dump(listofres, ff)
            
        
        
    #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #writer.writeheader()
    
    #if distance > 2.0:
        #print fitsimg
        #print sr[i]
        #print count

    #count += 1
    #DFF=pd.DataFrame(listofres[0:],columns=column_names)
    return #getdf(listofres)#listofres

  
if __name__ == "__main__":
    lc_annulus(IM)
#def getdf(listt):
#    DFFF=pd.DataFrame(listt[0:],columns=column_names)
#    return DFFF

#DFF=pd.DataFrame(listofres[0:],columns=column_names)
end=time.time()
print "duration:", end-start
