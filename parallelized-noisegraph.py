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
#fitslist=sorted(glob.glob(obs_dir+"CALed/"+obs_folder+"/"+"*.fits"))
fitslist=sorted(glob.glob(obs_dir+"CALed/AVERAGED-FINAL/"+obs_folder+"/"+"*S57*.fits"))
fitslist2=sorted(glob.glob(obs_dir+"CALed/AVERAGED-FINAL/"+obs_folder+"/"+"*S62*.fits"))
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
    #ranges=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,
    #        575,600,625,650,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1122]
    ranges=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,
            350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,
            680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,900,910,920,930,940,950,960,970,980,990,1000,
            1010,1020,1030,1040,1050,1060,1070,1080,1090,1100,1110,1120,1130,1139]
    #ranges2=[np.cos(np.radians(90))*1150,np.cos(np.radians(85))*1150,np.cos(np.radians(84))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,
    #        np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150,np.cos(np.radians(80))*1150]
    ranges2=ls
    #ranges=[0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500,525,550,
    #        575,600,625,650,700,725,750,775,800,825,850,875,900,925,950,975,1000,1025,1050,1075,1100,1122]
    
    
        #hdl=fits.open(i)
        #fitsimgdata= hdl[0].data[0,0,:,:]
    #imgdata=fits.getdata(img,header=False)[0,0,:,:]  # bug was here; no need to add dimensions # there is need otherwise ValueError: data must be a 2D array.
    #imgdata=fits.getdata(img,header=False)  #there is need otherwise ValueError: data must be a 2D array.
    imgdata=fits.getdata(img,header=False)[0,0,:,:] 

        
    resultlist = []

    for j in ranges:
        print "j is:"
        print j
        annulus_aperture = CircularAnnulus(position, r_in=j, r_out=j+10)
        
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
    #### THIS IS FOR SEEING RESULT I
    
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
    out2=Parallel(n_jobs=10,backend="multiprocessing", verbose=10)(delayed(noisedist)(i) for i in fitslist2)
    finalres=pd.DataFrame(out)
    finalres2=pd.DataFrame(out2)
    #finalres.to_pickle("/home/idayan/noisegraphdfLAST000Co202005051300.pkl") 
    finalres.to_pickle("/home/idayan/newnoisegrapDF/AVRimgs10inc57-{}.pkl".format(obs_folder))
    finalres2.to_pickle("/home/idayan/newnoisegrapDF/AVRimgs10inc62-{}.pkl".format(obs_folder))
    #finalres.to_pickle("/home/idayan/newnoisegrapDF/noisegraphdfLAST000Co-{}.pkl".format(obs_folder)) 
    #EDF.append(pd.DataFrame(out))
    #DF.append(pd.Series(out))
    #pd.DataFrame(out)

    #EDF=EDF.append(pd.DataFrame(pd.Series(out)))
    #EDF.append((out))
    
    
   
    end=time.time()
    print("processing time is:",end-start) 
