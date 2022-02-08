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



def get_configuration():
    """
    Returns a populated configuration
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default="./",
                        help="Input directory for fitsfile.")
    parser.add_argument('--fitsfile', type=str, default="./",
                        help="Target fits file.")

    parser.add_argument('--threshold', type=float, default=1000.0,
                        help="RMS Threshold to reject image.")
    parser.add_argument('--outdir', type=str, default="./",
                        help="Desitnation directory.")

    parser.add_argument("--detection", default=5, type=float,
                            help="Detection threshold")
    parser.add_argument("--analysis", default=3, type=float,
                            help="Analysis threshold")

    parser.add_argument("--radius", default=0, type=float,
                            help="Radius of usable portion of image (in pixels)")
    parser.add_argument("--grid", default=64, type=float,
                            help="Background grid segment size")

    parser.add_argument("--reference", default="", type=str,
                            help="Path of reference catalogue used for flux fitting. ")

    return parser.parse_args()

def process(cfg):
    
    
    #path, dirs, files = next(os.walk("/home/idayan"))
    path, dirs, files = next(os.walk(cfg.indir))
    fpath, fdirs, ffiles = (os.walk(cfg.indir))
    file_count = len(files)
    print file_count
    print files
    print dirs
    print path

if __name__ == "__main__":
    
    
        cfg = get_configuration()
        
        if cfg.outdir[-1] != "/":
            cfg.outdir = cfg.outdir+"/"

        if cfg.indir[-1] != "/":
            cfg.indir = cfg.indir+"/"

        if not os.path.isdir(cfg.outdir):
                os.mkdir(cfg.outdir)
        
        process(cfg)
