#import glob
#import astropy.io.fits as fits
#import numpy as np
#import pandas as pd
#from joblib import Parallel, delayed
#import os
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

imglst=glob.glob("/home/idayan/imglst/*.fits")

#len(imglst)

#print(len(imglst))

#imglst[0]
