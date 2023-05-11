from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u
from regions import CirclePixelRegion, PixCoord, CircleAnnulusPixelRegion
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import glob
import os
#from photutils import aperture
import pandas as pd
#import toolsHIGH

#import photutils
from photutils.aperture import CircularAperture
from photutils.aperture import CircularAnnulus
from photutils.aperture import aperture_photometry

#%matplotlib inline

import numpy.ma as ma

from photutils.datasets import make_100gaussians_image

from astropy.visualization import simple_norm


from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS



with open("/home/idayan/ALL-TXT/avrgucal202009290730.txt",'r') as f: # 12k
    lines=f.read().splitlines()
images = sorted(lines)
#images = images[4220:8440]
#images = images[6330:8440]
images = images[8440:10550]

srimages=pd.Series(images)
srimages_57_new=srimages[srimages.str.contains("-S57.0")]
srimages_62=srimages[srimages.str.contains("-S62.0")]



fitslist=srimages_57_new
fitsimg=fits.getdata(srimages_62.iloc[1])[:,:]



def rms(data):
    """Returns the RMS of the data about the median.
    Args:
        data: a numpy array
    """
    data -= np.median(data)
    return np.sqrt(np.power(data, 2).sum()/len(data))


def clipmy(data, sigma):
    """Remove all values above a threshold from the array.
    Uses iterative clipping at sigma value until nothing more is getting clipped.
    Args:
        data: a numpy array
    """
    data = data[np.isfinite(data)]
    raveled = data.ravel()
    median = np.median(data)
    std = np.std(data)
    #newdata = data[np.abs(data-median) <= sigma*std]
    #ma.masked_where(np.abs(data-median) <= sigma*std, data)
    #if len(newdata) and len(newdata) != len(raveled):
    newdata=np.where(np.abs(data-median) <= sigma*std, data, np.inf)
    #if len(newdata) and np.shape(newdata) != np.shape(data):
    if (newdata).size and np.count_nonzero(np.isinf(data)) != 0: 
        print(newdata)
        return clip(newdata, sigma)
    else:
        return newdata
    
def clip(data, sigma):
    """Remove all values above a threshold from the array.
    Uses iterative clipping at sigma value until nothing more is getting clipped.
    Args:
        data: a numpy array
    """
    data = data[np.isfinite(data)]
    raveled = data.ravel()
    median = np.median(raveled)
    std = np.nanstd(raveled)
    newdata = raveled[np.abs(raveled-median) <= sigma*std]
    if len(newdata) and len(newdata) != len(raveled):
        return clip(newdata, sigma)
    else:
        return newdata

import time
start=time.time()

position = [(np.shape(fitsimg)[0]/2., np.shape(fitsimg)[0]/2.)]
ranges=[0,100,200,300,400,500,600,700,800,900,1000]

#column_names=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10']
#df = pd.DataFrame(columns = column_names)
dff=pd.DataFrame()


#img_cntr=[1149,1149]
#radius_inner=0.1
#radius_outer=100
#area1pix=0.00612

for i in fitslist[0:90]:
    
    
        #hdl=fits.open(i)
        #fitsimgdata= hdl[0].data[0,0,:,:]
    #imgdata=fits.getdata(i,header=False)[0,0,:,:]
    imgdata=fits.getdata(i,header=False)[:,:]

        
    resultlist = []

    for j in ranges:
        annulus_aperture = CircularAnnulus(position, r_in=j, r_out=j+100)
        #skyregion = aperture.CircularAnnulus(position, r_in=j, r_out=j+100)
        #areaindeg = skyregion.area() * area1pix
    
        #annulus_aperture = CircularAnnulus(position, r_in=600, r_out=1100.)
        annulus_masks = annulus_aperture.to_mask(method='center')
        
        annulus_data=(annulus_masks[0].multiply(imgdata))
        #result=bkgrms(annulus_data)
        
        #masks = skyregion.to_mask(method='center')
        #annulus_data = masks[0].multiply(imgdata)
        #mask = masks[0].data
        mask = annulus_masks[0].data
        annulus_data_new = annulus_data[mask > 0]
        #annulus_data_new =annulus_data_new[(annulus_data_new != np.nan)& (annulus_data_new != 0)]
        
        
        #annulus_data_1d = annulus_data[mask > 0]
        #annulus_data_1d = annulus_data_1d[annulus_data_1d != 0]
        
        
        #_, median_sigclip, median_stdev = sigma_clipped_stats(annulus_data_1d, sigma=3)
        #_, median_sigclip, median_stdev = sigma_clipped_stats(annulus_data_new, sigma=3) ### this 4 time takes more time
        
        # ----- %%% -------
        # these are also giving same results
        
        # annulus_data2d=annulus_data[annulus_data != 0]
        # annulus_data2d=annulus_data[(annulus_data > 0) | (annulus_data < 0)]
        # result=rms(annulus_data2d)
        
        
        # ----- %%% --------
        
        #result=median_sigclip*areaindeg
        #resultt=(clip(annulus_data_1d, 3))
        #result=np.nanstd(resultt[resultt!=np.inf])
        #result= median_stdev
        #annulus_data_1d = annulus_data_1d[annulus_data_1d != np.inf]
        #sigmaclped_annulus=clip(annulus_data_1d,3)
        #result=rms(sigmaclped_annulus)
        
        #result=median_stdev
        result=rms(clip(annulus_data_new,3))
        
        print("result for j")
        print(result)
        #result.astype(pd.DataFrame)
        #pd.astype(result)
        #pd.DataFrame()
        #fresult=result.astype(float)
        resultlist.append(result)
        #pdresultlist=resultlist.tolist()
        
        
        print(j, 'j')

#pd.Series()       
    newrs=pd.Series(resultlist)

    #pd.Series(reesultlist)    
    dff=dff.append(newrs,ignore_index=True)
    dff.to_pickle("noisegraphdf-test62_2_test-202011080802.pkl")
    #dff.to_pickle("noisegraphdf-calw2ref-202006061630.pkl")
        
    #print "bottom i ", df
    

        #fitsimg.close()
        #hdl.close()
    
    
end=time.time()
print(end-start) 



dddf_62=pd.read_pickle("/home/idayan/JUPYTER/Article1/noisegraphdf-test62_gh1-202011080802.pkl")
dddf_57=pd.read_pickle("/home/idayan/JUPYTER/Article1/noisegraphdf-test57_gh1-202011080802.pkl")


