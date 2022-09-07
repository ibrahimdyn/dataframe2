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
#images = glob.glob(sys.argv[1])
images_folder = glob.glob(sys.argv[1]) # 202010030948
images_sub_folder = sorted(glob.glob(images_folder)) # ... _all
print images_sub_folder
images_2sub_folder=  sorted(glob.glob(images_sub_folder))  # SB... SB.ms...
print images_2sub_folder


