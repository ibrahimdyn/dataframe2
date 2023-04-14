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

#logging.basicConfig(level=logging.INFO)
#output_notebook()


#query_loglevel = logging.WARNING 

from scipy.optimize import curve_fit

from astropy.modeling.powerlaws import SmoothlyBrokenPowerLaw1D as SBPL


from scipy.stats import norm
from scipy.stats import halfnorm
from scipy.stats import chisquare
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import matplotlib
import matplotlib.gridspec as gridspec

import matplotlib.dates as mdates
matplotlib.use('Agg')



DF_target=pd.read_csv("/home/idayan/ALLFORCEDBlindDetsTAR1.csv")
DF_backg1=pd.read_csv("/home/idayan/ALLFORCEDBlindDetsBACKG1check.csv")
DF_target=DF_target[DF_target.freq_range<60]
DF_backg1=DF_backg1[DF_backg1.freq_range<60]
DF_target=DF_target[~DF_target.duplicated(['f_int','f_int_err','f_peak','det_sigma'])]
DF_backg1=DF_backg1[~DF_backg1.duplicated(['f_int','f_int_err','f_peak','det_sigma'])]
#DF_target=DF_target[~DF_target.taustart_ts.str.contains("2020-05-29")]
#DF_backg1=DF_backg1[~DF_backg1.taustart_ts.str.contains("2020-05-29")]
#DF_target=DF_target[~DF_target.taustart_ts.str.contains("2020-07-23")]
#DF_backg1=DF_backg1[~DF_backg1.taustart_ts.str.contains("2020-07-23")]

DF_target=DF_target[DF_target.taustart_ts.str.contains("2020-09-29")]
DF_backg1=DF_backg1[DF_backg1.taustart_ts.str.contains("2020-09-29")]



import datetime
roll_len = 1*40
roll_type = 'boxcar'
 

index_target = (( (DF_target.freq_range < 60.))).values

#index_target = (( (DF_target.freq_range < 60.) & 
#               (DF_target.rms_min < 12.00))).values

#index_target = (( (DF_target.freq_range > 60.) & 
#               ((DF_target.runcatid == 73) | (DF_target.runcatid == 379)) &
#               (DF_target.rms_min < 12.00))).values  # freq update #rms min remove

#index_target = (( (DF_target.freq_range > 6*1e7) & (DF_target.rms_min < 12.00) )).values  # freq update

#index_target = (((DF_target.runcatid == 73) | (DF_target.runcatid == 379) &
#         (DF_target.freq_range > 60.00) & (DF_target.rms_min < 12.00) )).values  # this works


index_backg1 = ( (DF_backg1.freq_range < 60.0) ).values  #rms update
#index_backg1 = ( (DF_backg1.freq_range < 60.0) & (DF_backg1.rms_min < 12.00)).values  #rms update

#index_backg1 = ( (DF_backg1.freq_range > 60.0)).values  
#index_backg1 = ((DF_backg1.freq_range > 6*1e7) & (DF_backg1.rms_min < 12.00)).values 

#index_backg1 = ((DF_backg1.freq_range > 60.00) & (DF_backg1.rms_min < 12.00) &
#             (DF_backg1.taustart_ts > datetime.datetime(2020, 7, 2, 0, 0, 0, 0))
#          &  (DF_backg1.taustart_ts < datetime.datetime(2020, 7, 5, 0, 0, 0, 0))).values 

_source_data = DF_target[index_target].set_index("taustart_ts") # coudlnt solvee the problem remove date_rannge

# no more pd_daterange # thee problem is prob bcs of irregular order of dates
# _source_data = DF_target[index_target].set_index("taustart_ts").loc[pd.date_range(start=np.min(DF_target[index_target].taustart_ts),
#                  end=np.max(DF_target[index_target].taustart_ts),
#                  freq="0.01S")] # coudlnt solvee the problem remove date_rannge

_source_data = _source_data.dropna()

rolling =_source_data.f_int.rolling(roll_len, win_type=roll_type) # should modify/update roll_type? for first nans

#flux_target =  _source_data.f_int 
flux_target = _source_data.f_int-rolling.mean().values
flux_target = flux_target[np.isfinite(flux_target)]

_source_data_backg1 = DF_backg1[index_backg1].set_index("taustart_ts")
##### update ! no more pd date range for the date of 202009
#_source_data_backg1 = DF_backg1[index_backg1].set_index("taustart_ts").loc[pd.date_range(start=np.min(DF_backg1[index_backg1].taustart_ts),
#                  end=np.max(DF_backg1[index_backg1].taustart_ts),
#                  freq="0.1S")]

_source_data_backg1 = _source_data_backg1.dropna()

rolling_backg1 =_source_data_backg1.f_int.rolling(roll_len, win_type=roll_type)

#flux_backg1 = _source_data_backg1.f_int
flux_backg1 = _source_data_backg1.f_int-rolling_backg1.mean().values
flux_backg1 = flux_backg1[np.isfinite(flux_backg1)]



plt.plot(_source_data.f_int, lw=0.3, label="B0950+08")
print "saving png file"
plt.savefig("/home/idayan/B0950+08_figs/testlc.png", bbox_inches = 'tight', pad_inches = 0)
