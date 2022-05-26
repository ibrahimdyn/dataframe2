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

#from photutils.datasets import make_100gaussians_image

#from astropy.visualization import simple_norm


from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS

from joblib import Parallel, delayed

import sys



obs_dir = "/zfs/helios/filer1/idayan/"
print "assigning sys obs argv1"
obs_folder= sys.argv[1]
print "assignned sys obs argv1", obs_folder


fitslist=sorted(glob.glob(obs_dir+"CALed/AVERAGED-FINAL/"+obs_folder+"/"+"*.fits"))
#print fitslist[0]
#print "header info"
#print fits.getdata(fitslist[0],header=False)[0,0,:,:]
print "printing len fitslist"
print len(fitslist)



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

print "sttartinng parrallel noise .py"

#column_names=['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10']
#df = pd.DataFrame(columns = column_names)
#dff=pd.DataFrame()


#img_cntr=[1149,1149]
#radius_inner=0.1
#radius_outer=100
#area1pix=0.00612
#def noisedist(imgs):
#for i in fitslist[0:100]:
#dff=pd.DataFrame()
def noisedist(img):
    #global dff
    position = [(1150., 1150.)]
    #ranges=[0,100,200,300,400,500,600,700,800,900,1000]
    #ranges=[0,50,100,150,200,250,300,350,400,450,500,550,
    #        600,650,700,750,800,850,900,950,1000,1050,1097]
    ranges=[0,50,100,150,200,250,300,350,400,450,500,550,
            600,650,700,750,800,850,900,950,1000,1050,1097]
    
    
        #hdl=fits.open(i)
        #fitsimgdata= hdl[0].data[0,0,:,:]
    #imgdata=fits.getdata(img,header=False)[0,0,:,:]  # bug was here; no need to add dimensions
    imgdata=fits.getdata(img,header=False)

        
    resultlist = []

    for j in ranges:
        annulus_aperture = CircularAnnulus(position, r_in=j, r_out=j+50)
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

        resultlist.append(result)
     
        newrs=pd.Series(resultlist)
    #print resultlist, type(resultlist)
    df = pd.DataFrame([resultlist])
    srs=pd.Series(resultlist)
    
    return srs
    #dff=dff.append(newrs, ignore_index=True)
    #dff.to_pickle("noisegraphdfLAST202006061630.pkl")
#dff=pd.DataFrame()
#1def integrate(img):
    
#1    global dff
 #1   dff=dff.append(noisedist(img), ignore_index=True)
  #1  dff.to_pickle("noisegraphdfLAST000202006061630.pkl") 
    #1return dff
    #dff.to_pickle("noisegraphdfLAST0202006061630.pkl") 
    #pd.concat(noisedist(img))
    #pd.concat([df, df2])
    #dfff.to_pickle("noisegraphdfLAST0202006061630.pkl") 
    #return dfff
#obs_dir = "/zfs/helios/filer0/idayan/"

#fitslist=sorted(glob.glob(obs_dir+"202006051431/"+"*_all*"+"/*SB*/"+"imgs/"+"2*.fits"))
#/zfs/helios/filer0/idayan/Cal60-202006061630/
#fitslist=sorted(glob.glob(obs_dir+"CALed/"+obs_folder+"/"+"*.fits"))
print "first of fitlisst:"


if __name__ == "__main__":
    #[(integrate(i)) for i in fitslist[0:3]]
    EDF=pd.DataFrame
    out=Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(noisedist)(i) for i in fitslist)
    finalres=pd.DataFrame(out)
    #finalres.to_pickle("/home/idayan/noisegraphdfLAST000Co202005051300.pkl") 
    finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfAVRGF-{}.pkl".format(obs_folder))
    #finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfLAST000Co-{}.pkl".format(obs_folder)) 
    #EDF.append(pd.DataFrame(out))
    #DF.append(pd.Series(out))
    #pd.DataFrame(out)

    #EDF=EDF.append(pd.DataFrame(pd.Series(out)))
    #EDF.append((out))
    
    
   
    end=time.time()
    print("processing time is:",end-start) 
