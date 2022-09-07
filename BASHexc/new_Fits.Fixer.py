#!/bin/python 


from astropy.io import fits
import glob
import os
import time 
import sys
from joblib import Parallel, delayed

print "AAAA"
print sys.argv[1]
print "BBB"
A=sys.argv[1]
#images = glob.glob(sys.argv[1])

images=glob.glob("/zfs/helios/filer1/idayan/"+ str(A) + "/*_all" + "/SB*" + "/imgs/*.fits" )

def fix_fits(img):
    
    hdulist = fits.open(img, mode='update')
    prihdr = hdulist[0].header

    # Don't know which of these is needed
    prihdr.update({'RESTFRQ':hdulist[0].header['CRVAL3']})
    prihdr.update({'RESTFREQ':hdulist[0].header['CRVAL3']})
    
    prihdr.update({'RESTBW':hdulist[0].header['CDELT3']})
    
    # These should be fit/written by WSClean
    prihdr.update({'BMIN':0.2})
    prihdr.update({'BMAJ':0.2})
    

    if hdulist[0].header['CRVAL1']<0.:
        ra=hdulist[0].header['CRVAL1']+360.
        prihdr['CRVAL1']=ra
    
    hdulist.flush()
    
    if (os.path.basename(img)[:3] == "000") or (os.path.basename(img)[:2] == "00"):
        newname = os.path.dirname(img)\
        +"/"+str(hdulist[0].header["DATE-OBS"])\
        +"-"+os.path.basename(img)[6:11]+".fits"

        os.rename(img, newname)

_out = Parallel(n_jobs=12)(delayed(fix_fits)(img) for img in images)
