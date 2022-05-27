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


#fitslist=sorted(glob.glob(obs_dir+"CALed/AVERAGED-FINAL/"+obs_folder+"/"+"*.fits")) # for crrection remove AVEERAGE --averaged files gettig flatter
fitslist=sorted(glob.glob(obs_dir+"CALed/"+obs_folder+"/"+"*.fits"))
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

ls=[]
for i in range(133):
    #100*np.sqrt(100)
    print 10*np.sqrt(i)
    ls.append(100*np.sqrt(i))
print "len of ls is"
print len(ls)

def noisedist2(img):
    #global dff
    position = [(1150., 1150.)]
    
    #ranges=[0,100,200,300,400,500,600,700,800,900,1000]
    #ranges=[0,50,100,150,200,250,300,350,400,450,500,550,
    #        600,650,700,750,800,850,900,950,1000,1050,1097]
    #ranges=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,
    #        575,600,625,650,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1122]
    #ranges=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,
    #        350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,
    #        680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000,
    #        1010,1020,1030,1040,1050,1060,1070,1080,1090,1100,1110,1120,1130,1139]
    #ranges2=[np.cos(np.radians(90))*1150,np.cos(np.radians(85))*1150,np.cos(np.radians(84))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150]
    ranges2=ls
    #ranges=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,
    #        575,600,625,650,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1122]
    
    
    imgdata=fits.getdata(img,header=False)[0,0,:,:] 

        
    #resultlist = []
    #index=0
    resultlist2=[]
    for ind in range(len(ranges2)):
        
        print "ind is:"
        print ind
        print ranges2[ind]
        annulus_aperture2 = CircularAnnulus(position, r_in=ranges2[ind], r_out=ranges2[ind+1])

        annulus_masks2 = annulus_aperture2.to_mask(method='center')

        annulus_data2=(annulus_masks2[0].multiply(imgdata))

        mask2 = annulus_masks2[0].data
        annulus_data_new2 = annulus_data[mask2 > 0]

        result2=rms(clip(annulus_data_new2,3))

        print("result for j")
        print(result2)

        resultlist2.append(result2)

        newrs2=pd.Series(resultlist2)
        #index +=1
    #print resultlist, type(resultlist)
    #df = pd.DataFrame([resultlist]) # absol
    srs2=pd.Series(resultlist2)

    return srs2

 
print "first of fitlisst:"


if __name__ == "__main__":
    #[(integrate(i)) for i in fitslist[0:3]]
    EDF=pd.DataFrame
    out=Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(noisedist2)(i) for i in fitslist)
    finalres=pd.DataFrame(out)
    #finalres.to_pickle("/home/idayan/noisegraphdfLAST000Co202005051300.pkl") 
    finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfAVRGF-NITcrEQAREA-{}.pkl".format(obs_folder))
    #finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfLAST000Co-{}.pkl".format(obs_folder)) 
    #EDF.append(pd.DataFrame(out))
    #DF.append(pd.Series(out))
    #pd.DataFrame(out)

    #EDF=EDF.append(pd.DataFrame(pd.Series(out)))
    #EDF.append((out))
    
    
   
    end=time.time()
    print("processing time is:",end-start) 
