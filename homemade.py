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


    '''
    try:
        isZEA = filepath.endswith('.fits') and any(['ZEA' in x.upper() for x in [hdu.header['CTYPE1'], hdu.header['CTYPE2']] ])
    except:
        #CTYPE1 or CTYPE2 not found in header keys
        #Assume file must not be ZEA... Treat as a rectangle
        isZEA = False

    if isZEA:
        #data = np.where(np.isfinite(data), data, np.zeros_like(data))
        #plt.contour(data)
        #plt.show()
        #plt.imshow(data)
        #plt.show()
        #exit()

        #Making Round Mask for ZEA data 
        datay, datax = data.shape
        mnvals = np.indices(data.shape)
        pixcrd = np.zeros((datax*datay,2), np.float_)
        pixcrd[:,0] = mnvals[:,:][0].reshape(datax*datay)
        pixcrd[:,1] = mnvals[:,:][1].reshape(datax*datay)

        #Use astropy.wcs module to transform coords
        w = wcs.WCS(hdu.header) #TODO verify accuracy of this_______________________________________________________
        #w = wcs.WCS(naxis=2)
        #w.wcs.crpix = [hdu.header['CRPIX1'], hdu.header['CRPIX2']] #[1.125000000E3, 1.125000000E3]
        #w.wcs.cdelt = [hdu.header['CD1_1'], hdu.header['CD2_2']] #np.array([-8.00000000E-2, 8.00000000E-2])
        #w.wcs.crval = [hdu.header['CRVAL1'], hdu.header['CRVAL2']]#[0.00000000E0, -9.00000000E1]
        #w.wcs.ctype = [hdu.header['CTYPE1'], hdu.header['CTYPE2']] #['RA---ZEA', 'DEC--ZEA']
        
        worldc = w.wcs_pix2world(pixcrd, 1)
        worldcra = worldc[:,0].reshape(*data.shape) 
        worldcdec = worldc[:,1].reshape(*data.shape)
        #plt.contour(worldcra)
        #plt.show()
        #plt.contour(worldcdec)
        #plt.show()
        
        #Mask at both diameters: 2*smr and wlen
        gm = np.zeros_like(data)
        wsquare1 = np.ones((2*smr, 2*smr), np.int) 
        wkernel = circ_kern(wsquare1, 2*smr)
        gm[worldcdec < 0] = 1
        gmconv = scipy.ndimage.filters.correlate(gm, weights=wkernel)
        gg = gmconv.copy()
        gg[gmconv < np.max(gmconv)] = 0
        gg[gmconv == np.max(gmconv)] = 1
        smr_mask = gg

        gm = np.zeros_like(data)
        wsquare1 = np.ones((wlen, wlen), np.int) 
        wkernel = circ_kern(wsquare1, wlen)
        gmconv = scipy.ndimage.filters.correlate(gm, weights=wkernel)
        gm[worldcdec < 0] = 1
        gg = gmconv.copy()
        gg[gmconv < np.max(gmconv)] = 0
        gg[gmconv == np.max(gmconv)] = 1
        wlen_mask = gg

        #TODO___ TRANSFORM GOES UNUSED???______________________________________________________________________________
        #return data, smr_mask, wlen_mask
    '''


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

