import os
import sys
import csv

from datetime import datetime
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

import logging
logging.basicConfig(level=logging.INFO)

import h5py
from scipy import interpolate

# In[ ]:
obs=(sys.argv[1]).split("/")[5]
#fitsf=str(sys.argv[1:])
#--fitsfile=/zfs/helios/filer1/idayan/202005051300/2020-05-05T13:02:02-13:05:12_all/SB281-2020-05-05T13:02:02-13:05:12/imgs/2020-05-05T13:02:02.0-SB281.fits/

def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()
    #/zfs/helios/filer0/idayan/Partial60Mhzcal-test
    
    parser.add_argument('--indir', type=str, default="",
                        help="Input directory for fitsfile.")
    #parser.add_argument('--fitsfile', type=str, default="".format(fitsf),
    #                    help="Target fits file.")
    parser.add_argument('--fitsfile', type=str, default="",
                        help="Target fits file.")

    #parser.add_argument('--indir', type=str, default="/zfs/helios/filer1/idayan/",
    
    #                    help="Input directory for fitsfile.")
    #parser.add_argument('--fitsfile', type=str, default="/zfs/helios/filer1/idayan/folder/*_all/\
    #SB*/imgs/*.fits",
    #                    help="Target fits file.")
    
    #parser.add_argument('--fitsfile', type=str, default="./",
    #                    help="Target fits file.")

    parser.add_argument('--threshold', type=float, default=1000.0,
                        help="RMS Threshold to reject image.")
    parser.add_argument('--outdir', type=str, default="/zfs/helios/filer1/idayan/UCALED/{}/".format(obs),
                        help="Desitnation directory.")
    
    #parser.add_argument('--outdir', type=str, default="./",
    #                    help="Desitnation directory.")

    parser.add_argument("--detection", default=3, type=float,
                            help="Detection threshold")
    parser.add_argument("--analysis", default=3, type=float,
                            help="Analysis threshold")

    parser.add_argument("--radius", default=0, type=float,
                            help="Radius of usable portion of image (in pixels)")
    parser.add_argument("--grid", default=64, type=float,
                            help="Background grid segment size")
    
    parser.add_argument("--reference", default="/home/idayan/AARTFAAC_catalogue.csv", type=str,
                            help="Path of reference catalogue used for flux fitting. ")

    #parser.add_argument("--reference", default="", type=str,
    #                        help="Path of reference catalogue used for flux fitting. ")

    return parser.parse_args()

def get_beam(freq):
    beams = np.array([30,35,40,50,55,60,65,70,75,80,85,90])
    freq_to_use = str(beams[np.argsort(np.abs(freq-beams))[0]])

    beam_file = "/home/idayan/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_{}MHz.hdf5".format(freq_to_use)
    #beam_file = "/home/mkuiack1/AARTFAAC_beamsim/LBAOUTER_AARTFAAC_beamshape_{}MHz.hdf5".format(freq_to_use)
    orig =  np.array(h5py.File(beam_file, mode="r").get('lmbeamintensity_norm'))

    # Make its coordinates; x is horizontal.
    x = np.linspace(0, 2300, orig.shape[1])
    y = np.linspace(0, 2300, orig.shape[0])

    # Make the interpolator function.
    f = interpolate.interp2d(x, y, orig, kind='linear')

    # Construct the new coordinate arrays.
    x_new = np.arange(0, 2300)
    y_new = np.arange(0, 2300)

    # Do the interpolation.
    return f(x_new, y_new)


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


# In[ ]:


def compare_flux(sr, catalog_ras, catalog_decs, catalog_fluxs, catalog_flux_errs):
    '''
    Compares the two catalogues, matching sources, and outputs the results of linear fit to the fluxes. 
    '''
    x = []
    y = []
    w = []
    sr_indexes = []
    cat_indexes = []
    
    #distances_2 = []
    #sr_err = []
    #cat_err =[]


    for i in range(len(sr)):
        #print sr[i].flux_value
        
        if (sr[i].flux.value>5) & (sr[i].flux.value<1600):
            

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

        #if (len(x) > 2) and (x[0]-x[1]!=0):
        # slope cor reaches 1e9, remove second condition
    if (len(x) > 2):
	
        w = np.array(w,dtype=float)
        fit,cov = np.polyfit(x,y,1,w=1./w,cov=True)
    else:
	
	fit = [1e9,1e9,1e9,1e9,0]
	cov = np.array([[1e9, 1e9], [1e9, 1e9]])

        #return fit[0], cov[0,0], fit[1], cov[1,1], len(x)
    return fit[0], fit[1], x, y, cat_indexes

    #if len(x) > 2 and (x[0]-x[1]!=0):       
    #    w = np.array(w,dtype=float)
    #    fit,cov = np.polyfit(x,y,1,w=1./w,cov=True)
    #return fit[0], cov[0,0], fit[1], cov[1,1], len(x)
    #else:
    #    return 1e9, 1e9, 1e9, 1e9, 0
#    return fit[0], cov[0,0], fit[1], cov[1,1], len(x)


def process(cfg):
    '''
    Perform an initial quality control filtering step on the incoming image stream. Images
    which are not rejected are then flux calibrated using a reference catalogue.
    '''
    #ref_cat = pd.read_csv("/home/idayan/AARTFAAC_catalogue.csv")
   
    print "running process"

    lofarfrequencyOffset = 0.0
    lofarBW = 195312.5
    
    ref_cat = pd.read_csv(cfg.reference)
    
    fitsimg = fits.open(cfg.indir+cfg.fitsfile)[0]
    
    t = Time(fitsimg.header['DATE-OBS'])
    frq = fitsimg.header['RESTFRQ']
    bw = fitsimg.header['RESTBW']


    # Initial quality condition. 
    #if np.nanstd(fitsimg.data[0,0,:,:]) < cfg.threshold:
    if  (np.nanstd(fitsimg.data[0,0,:,:]) < cfg.threshold) and (np.nanstd(fitsimg.data[0,0,:,:]) > 2) :
	
        flux_compare=[]
	
	
	# APPLY BEAM CORRECTION, then check quality with flux scaling
	#bg_data, bg_f =fits.getdata(img, header=True)
	print "PRINTING IMAGE"
	print cfg.fitsfile
	bg_data, bg_f =fits.getdata(cfg.fitsfile, header=True)
    	beam_model = get_beam(bg_f["CRVAL3"]/1e6)
	fitsimg.data[0,0,:,:] = fitsimg.data[0,0,:,:]*(np.max(beam_model)/beam_model)

        # Source find 
        configuration = {
            "back_size_x": cfg.grid,
            "back_size_y": cfg.grid,
            "margin": 0,
            "radius": cfg.radius}

        img_HDU = fits.HDUList(fitsimg)
        imagedata = sourcefinder_image_from_accessor(open_accessor(fits.HDUList(fitsimg),
                                                                   plane=0),
                                                     **configuration)

        sr = imagedata.extract(det=cfg.detection, anl=cfg.analysis,
                               labelled_data=None, labels=[],
                               force_beam=True)

        # Reference catalogue compare
        slope_cor, intercept_cor, ref_match, image_match, index_match= compare_flux(sr,
                                       ref_cat["ra"],
                                       ref_cat["decl"],
                                       ref_cat["f_int"],
                                       ref_cat["f_int_err"])
	
	flux_compare.append([np.array(ref_match),
                           # (np.array(image_match))/slope_cor,
                            (np.array(image_match) - intercept_cor)/slope_cor,
                     #(np.array(image_match)* slope_cor ) + intercept_cor,
                            np.array(np.ravel(index_match)),np.array(image_match)])


        #slope_cor, intercept_cor, ref_match, image_match, index_match, DST2,SR_err,CAT_err
        #slope_cor,slope_err, intercept_cor, int_err, N_match  = compare_flux(sr,
        #                               ref_cat["ra"],
        #                               ref_cat["decl"],
        #                               ref_cat["f_int"],
        #                               ref_cat["f_int_err"])

        #fields=[slope_cor,slope_err, intercept_cor,int_err, N_match]
        
        test = pd.DataFrame([])
        

        for i in range(len(flux_compare)):
            
            if len(test) == 0:
			
			
                
			test = pd.DataFrame({"reference":flux_compare[i][0],
					   "image":flux_compare[i][1],
					  "IMnorm":flux_compare[i][3]},
					  index=flux_compare[i][2])

            else:
		
                test = pd.concat([test,
                         pd.DataFrame({"reference":flux_compare[i][0],
                                       "image":flux_compare[i][1],
                                      "IMnorm":flux_compare[i][3]},
                                      index=flux_compare[i][2])])

        _STD_=np.std(test.image/test.reference)
        
	fields=[slope_cor, intercept_cor, ref_match, image_match, index_match, _STD_ ]
        #fields=[slope_cor,slope_err, intercept_cor,int_err, N_match, _STD_]
	#slope_cor, intercept_cor, ref_match, image_match, index_match
	
	#202009290730
	#202005181400
    #ALLimgpathstofluxcal1-202005052000.txt

        #with open(r'/home/idayan/fit_results_051217.csv', 'a') as f:
        #with open(r'/home/idayan/fit_results_202006041232.csv', 'a') as f:
 	#with open(r'/home/idayan/fit_results_202006051232.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202009290730.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010030948.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010031118.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010131201.csv', 'a') as f:
	#202010201005
	#with open(r'/home/idayan/fit_results_202102181807.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010201005.csv', 'a') as f: # 1130 was written to here
	#with open(r'/home/idayan/fit_results_202010031250.csv', 'a') as f:
	
	#with open(r'/home/idayan/fit_results_202006051431.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010131021.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202010060710.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202011021014.csv', 'a') as f:
	#with open(r'/home/idayan/fit_results_202011080802_2_rmnng80k.csv', 'a') as f:
	
	#with open(r'/home/idayan/fit_results_202006041232.csv', 'a') as f:
	with open(r'/home/idayan/fit_results_202011100907.csv', 'a') as f:
	
	#with open(r'/home/idayan/fit_results_202006061232.csv', 'a') as f:
		
	
	
	
	
		
	#tocalqualNEW-202010031118.txt
	#tocalqualNEW-202010030948.txt
        
		
		
    #with open(r'/home/idayan/fit2_202005181400_results.csv', 'a') as f:
    #with open(r'/home/idayan/fit2_202009290730_results.csv', 'a') as f:
  #with open(r'/home/idayan/fit2_results.csv', 'a') as f:
		writer = csv.writer(f)
		writer.writerow(fields)

        # Slope set to 1e9 if line fit fails
        # Slope set to 1e9 if line fit fails AND std of ratios excess some threshold say 0.45
	
        #if slope_cor < 1e8:
	# try with 0.45; for the date 202006041232 only 1 percent pass the 0.40 std filter
        if (slope_cor < 1e8) and ( _STD_ < 0.45): # for the date 202006041232 only 1 percent pass the 0.40 std filter
	        filename = '%s.fits' % (datetime.fromtimestamp(t.unix).strftime('%Y-%m-%dT%H:%M:%S')+ \
		    "-S"+str(round((frq-lofarfrequencyOffset)/lofarBW,1))+ \
		    "-B"+str(int(np.ceil(bw /lofarBW))))
		
		# APPLY BEAM CORRECTION 
		#fitsimg.data[0,0,:,:] = fitsimg.data[0,0,:,:]*(np.max(beam_model)/beam_model)

		fitsimg.data[0,0,:,:] = (fitsimg.data[0,0,:,:]-intercept_cor)/slope_cor
		#            fitsimg.writeto(cfg.outdir+filename,overwrite=True)
		os.remove(cfg.indir+cfg.fitsfile)
		print "writing", cfg.outdir+filename
		fitsimg.writeto(cfg.outdir+filename,overwrite=True)
        else:
		
       		print "slope fail", slope_cor, "<", "1e8"
		#print
		print "STD 1 is:", _STD_ 
            	os.remove(cfg.indir+cfg.fitsfile)
                return
    else:
	
	
        print "image QC fail", np.nanstd(fitsimg.data[0,0,:,:]), "<", cfg.threshold
	#there is no _STD_ here!!!
        #print "STD 2 is:", _STD_ 
        os.remove(cfg.indir+cfg.fitsfile)
        return


if __name__ == "__main__":
    
    
        cfg = get_configuration()
        
       # if cfg.outdir[-1] != "/":
       #     cfg.outdir = cfg.outdir+"/"

        #if cfg.indir[-1] != "/":
        #    cfg.indir = cfg.indir+"/"

        if not os.path.isdir(cfg.outdir):
                os.mkdir(cfg.outdir)
        
        process(cfg)
sys.exit()

