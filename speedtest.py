#!/usr/bin/python

#Lowell Schudel
#Execute 'python speedtest.py'
#speedtest.py compares fastrht_prof.py and fastrhtOLD_prof.py
#Both will use a single test image, smalltest.fits
#Both will use the speedtest.getData() and speedtest.setParams() function
#Neither will write their output to file

#Imports
import timeit
import numpy as np
import math
import copy
from astropy.io import fits


#Shared Definitions
def getData(filepath):
    if filepath.endswith('.fits'):
        hdulist = fits.open(filepath) #Opens HDU list
        gassslice = hdulist[0].data #Reads all data as an array
    else:
        gassslice = imread(filepath, True)[::-1] #Makes B/W array, reversing y-coords
    x, y = gassslice.shape #Gets dimensions
    return gassslice, x, y

def circ_kern(inkernel, radius):
    #Performs a circle-cut of given radius on inkernel.
    #Outkernel is 0 anywhere outside the window.    
    #These are all the possible (m,n) indices in the image space, centered on center pixel
    mnvals = np.indices((len(inkernel), len(inkernel)))
    kcntr = np.floor(len(inkernel)/2.0)
    mvals = mnvals[:,:][0] - kcntr
    nvals = mnvals[:,:][1] - kcntr

    rads = np.sqrt(nvals**2 + mvals**2)
    outkernel = copy.copy(inkernel)
    outkernel[rads > radius/2] = 0
    
    return outkernel

def setParams(gassslice, w, s, f):
    wlen = w #101.0 #Window diameter
    frac = f #0.70 #Theta-power threshold to store
    smr = s #11.0 #Smoothing radius

    ulen = np.ceil(wlen + smr/2) #Must be odd
    if np.mod(ulen, 2) == 0:
        ulen += 1
    ucntr = np.floor(ulen/2)

    wcntr = np.floor(wlen/2)
    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))  

    theta, dtheta = np.linspace(0.0, np.pi, ntheta, endpoint=False, retstep=True)
    
    wsquare1 = np.ones((wlen, wlen), np.int_)
    kernel = circ_kern(wsquare1, smr) 
    wkernel = circ_kern(wsquare1, wlen) 
    
    mask = None #Default is no mask

    return wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask

if __name__ == '__main__':

    #Setup Statements
    N=3
    SETUP = 'from speedtest import getData, setParams; gassslice, datay, datax = getData(\'smalltest.fits\'); wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask = setParams(gassslice, 55, 5, 0.70)'

    #Timing Loop
    newstmt = 'fastrht_prof.window_step(gassslice, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask)'
    newsetup = 'import fastrht_prof; '+SETUP
    newtime = timeit.timeit(stmt=newstmt, setup=newsetup, number=N)/N
    print 'NEW:', newtime

    oldstmt = 'fastrhtOLD_prof.window_step(gassslice, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask)' 
    oldsetup = 'import fastrhtOLD_prof; '+SETUP
    oldtime = timeit.timeit(stmt=oldstmt, setup=oldsetup, number=N)/N
    print 'OLD:', oldtime
    
    print 'SPEEDUP:' + str(100*(1.0-newtime/oldtime)) + '%'
    print 'Done.'


'''
Result History:
6/16/14: old=112.19, new=111.97

'''


