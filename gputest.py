#!/usr/bin/python

#Lowell Schudel
#GPU Test

import numpy as np
from numbapro import autojit
import timeit
import copy
#_______________________________________________

def function(img, theta=None, idl=False):
    if img.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    
    wx, wy = img.shape    
    wmid = np.floor(wx/2)
    
    if idl:
        ntheta = math.ceil((np.pi*np.sqrt(2)*((wx-1)/2.0)))  
        theta = np.linspace(0, np.pi, ntheta)

    # compute the vertical bins (the distances)
    d = np.ceil(np.hypot(*img.shape))
    nr_bins = d
    bins = np.linspace(-d/2, d/2, nr_bins)

    # allocate the output image
    out = np.zeros((nr_bins, len(theta)), dtype=np.uint64)

    # precompute the sin and cos of the angles
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # find the indices of the non-zero values in
    # the input image
    y, x = np.nonzero(img)

    # x and y can be large, so we can't just broadcast to 2D
    # arrays as we may run out of memory. Instead we process
    # one vertical slice at a time.
    for i, (cT, sT) in enumerate(zip(cos_theta, sin_theta)):

        # compute the base distances
        distances = (x - wmid) * cT + (y - wmid) * sT
        #distances = np.add(np.multiply(x, cT), np.multiply(y, sT)) - wmid*(cT+sT) 
        
        # round the distances to the nearest integer
        # and shift them to a nonzero bin
        #shifted = np.round(distances) - bins[0]

        # cast the shifted values to ints to use as indices
        #indices = shifted.astype(np.int)
        
        # use bin count to accumulate the coefficients
        #bincount = np.bincount(indices)
        bincount = np.bincount(np.subtract(np.round(distances), bins[0]).astype(np.int), minlength=nr_bins)
        
        # finally assign the proper values to the out array
        #out[:len(bincount), i] = bincount

        out.T[i] = bincount
        #for j in np.arange(bincount.shape[0]):
            #out.T[i][j] = bincount[j]

    return out, theta, bins


if __name__ == '__main__':

    #Setup Statements
    N=10
    SETUP = 'import numpy as np; img = np.ones((600, 600)); from gputest import function'

    #Timing Loop
    oldstmt = 'function(img)' 
    oldtime = timeit.timeit(stmt=oldstmt, setup=SETUP, number=N)/N
    print 'OLD:', oldtime

    newstmt = 'gu_func(img)'
    newsetup = SETUP + '; from numbapro import autojit; gu_func = autojit(target="cpu")(function)'
    newtime = timeit.timeit(stmt=newstmt, setup=newsetup, number=N)/N
    print 'NEW:', newtime

    
    print 'SPEEDUP:' + str(100*(1.0-newtime/oldtime)) + '%'
    print 'Done.'

