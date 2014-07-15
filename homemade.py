#!/usr/bin/python
#FAST ROLLING HOUGH TRANSFORM
#Susan Clark, Lowell Schudel

#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import scipy.ndimage
import math
from astropy import wcs
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import sys
import string

import rht

#-----------------------------------------------------------------------------------------
#Functions
#-----------------------------------------------------------------------------------------
def mega_rht(path, force=force, wlen=wlen, frac=frac, smr=smr):
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    output='.'
    xyt_filename = os.path.join(output, filename + '_xyt.npz')

    if not force and os.path.isfile(xyt_filename):
        return True
    
    #TODO START HERE____________________________________________________________________________________
    xy_array, mask = getData(filepath, make_mask=True, wlen=wlen) 
    datay, datax = xy_array.shape
    print '1/4:: Analyzing', filename, str(datax)+'x'+str(datay)

    #TODO wrap parameter input
    wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, bad_mask = setParams(wlen, smr, frac)
    print '2/4:: Window Diameter:', str(wlen)+',', 'Smoothing Radius:', str(smr)+',', 'Threshold:', str(frac)

    hthets, hi, hj = window_step(xy_array, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask) #TODO theta, ntheta, mask
    
    putXYT(xyt_filename, hi, hj, hthets)

    print '4/4:: Successfully Saved Data As', xyt_filename
    return True



if __name__ == "__main__":
    
    source = sys.argv[1]
    args = sys.argv[2:]
    
    #Default flag values
    DISPLAY = False
    FORCE = False
    
    #Default param values
    wlen = rht.WLEN
    frac = rht.FRAC
    smr = rht.SMR

    for arg in args:
        if '=' not in arg:
            #FLAGS which DO NOT carry values 
            if arg.lower() in ['d', '-d', 'display', '-display' ]:
                DISPLAY = True
            elif arg.lower() in ['f', '-f', 'force', '-force' ]:
                FORCE = True
        else:
            #PARAMETERS which DO carry values
            argname = arg.lower().split('=')[0]
            argval = arg.lower().split('=')[1] #TODO Handle errors
            if argname in ['w', 'wlen', '-w', '-wlen']:
                wlen = float(argval)
            elif argname in ['s', 'smr', '-s', '-smr']:
                smr = float(argval)
            elif argname in ['f', 'frac', '-f', '-frac']:
                frac = float(argval)

    #Interpret whether the Input is a file or directory, excluding all else
    pathlist = []
    if os.path.isfile(source):
        pathlist.append(source)
    elif os.path.isdir(source):
        for obj in os.listdir(source):
            obj_path = os.path.join(source, obj)
            if os.path.isfile(obj_path):
                pathlist.append(obj_path)
    pathlist = filter(rht.is_valid_file, pathlist)


    #Run RHT Over All Valid Inputs 
    announce(['Fast Rolling Hough Transform by Susan Clark', 'Started for: '+source])
    #TODO batch progress bar
    summary = []
    for path in pathlist:
        success = True
        try:
            success = mega_rht(path, force=force, wlen=wlen, frac=frac, smr=smr)
            if (display):
                success &= rht.viewer(path, force=False, wlen=wlen, frac=frac, smr=smr)
        except:
            success = False
            raise #TODO _______________________________________________________________________Messy failure
        finally:
            if success:
                summary.append(path+': Passed')
            else:
                summary.append(path+': Failed')
    summary.append('Complete!')
    announce(summary)
    exit()

