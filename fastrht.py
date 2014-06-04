#!/usr/bin/python

# FAST ROLLING HOUGH TRANSFORM

# wlen : Diameter of a 'window' - the data to be evaluated at one time
# frac : fraction (percent) of one angle that must be 'lit up' to be counted
# smr  : smoothing radius of unsharp mask function.
# ulen : length of unsharp mask square. Must be at least wlen + smr/2

# ---------------------------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import math
import time
import random
from astropy import wcs
from astropy.io import fits
import scipy as sp
from scipy.ndimage import filters
#import pyfits
import copy

'''
# Load in data, set any corrupted data (nans, -999s, etc) to None value. 
# This is currently a bunch of fnum if-statements: this was to convenience my switching between data sets.
# Clearly getData is path-dependent.

def getData(first, last, fnum):
    if fnum == 0:
       # gassfile = '/Users/susanclark/Documents/gass_10.zea.fits'
        gassfile = '/share/galfa/gass_10.zea.fits'
    if fnum == 1:
        #gassfile = '/Users/susanclark/Documents/gass_11.zea.fits'
        gassfile = '/share/galfa/gass_11.zea.fits'
    if fnum == 2:
        gassfile = '/home/goldston/Download/gass_11_ra270_dec0_ch0-9.zea.fits'

    if fnum < 4:
        gassdata  = pyfits.getdata(gassfile, 0)
        gassslice = np.sum(gassdata[first:last, :, :], axis=0)

    if fnum == 4:
    
        gassdata0 = pyfits.getdata('/share/galfa/gass_10.zea.fits', 0)
        gassslice0= np.sum(gassdata0[first:62, :, :], axis=0)

        gassdata1 = pyfits.getdata('/share/galfa/gass_11.zea.fits', 0)
        gassslice1 = np.sum(gassdata1[0:last, :, :], axis=0)

        gassslice = gassslice1 + gassslice0
        
    if fnum == 5:
        hdulist = fits.open('/share/galfa/GC.hi.tb.allgal.fits')
        gassslice = hdulist[0].data
        gassslice = np.sum(gassslice[0,first:last, :, :], axis=0)*(-1)
        
    if fnum == 6:
        hdulist = fits.open('/share/galfa/destripe_zenith_WRONG_NAXIS3.fits')
        gassslice = hdulist[0].data
        gassslice = gassslice[last, :, 0:4501]
        gassslice[gassslice < 0] = None
        
    if fnum == 7:
        hdulist = fits.open('/share/galfa/destripe_zenith_WRONG_NAXIS3.fits')
        gassslice = hdulist[0].data
        gassslice = gassslice[last, :, 4300:]    
        gassslice[gassslice < 0] = None    
        
    if fnum == 8:
        hdulist = fits.open('/share/galfa/GC.hi.tb.allgal.fits')
        gassslice = hdulist[0].data
        gassslice[gassslice > 120] = None
        gassslice[gassslice < 0] = None
        gassslice = gassslice[0,last, :, :]*(-1)
        
    datay, datax = np.shape(gassslice)
    print datay, datax
    
    return gassslice, datay, datax

'''
#------------------------------ Ok with this >
def getData(filename):
    #This could replace the specialized code above if I'm using simpler fits images
    hdulist = fits.open(filename) #Opens HDU list
    gassslice = hdulist[0].data #Reads all data as an array
    x, y = gassslice.shape #Gets dimensions
    return gassslice, x, y

def setParams(gassslice, w, s, f, gass=False):
    wlen = w #101.0 #Window diameter
    frac = f #0.70 #Theta-power threshold to store
    smr = s #11.0 #Smoothing radius
    #------------------------------ < Until Here.
	
	#Here in setParams, I'm not sure why ntheta is picked.
	#Also, I don't know what ucenter gets used for.
    var = 'test'
    ulen = np.ceil(wlen + smr/2) #Must be odd
    
    if np.mod(ulen, 2) == 0:
        ulen += 1
    ucntr = np.floor(ulen/2)
    wcntr = np.floor(wlen/2)

    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))  

    #------------------------------ Ok with this >
    dtheta = np.pi/ntheta
    theta = np.arange(0, np.pi, dtheta)
    
    wsquare1 = np.ones((wlen, wlen), np.int_)
    kernel = circ_kern(wsquare1, smr) 
    wkernel = circ_kern(wsquare1, wlen) 
    
    if gass==True:
        mask = makemask(wkernel, gassslice)
    else:
        mask = None #Default is no mask

       # xyt = np.load('xyt2_101_223.npy')
       # mask = np.load('w101_mask.npy')

    return wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask

#------------------------------ < Until Here.

'''
# The following is specific to a certain data set (the Parkes Galactic All-Sky Survey)
# which was in a Zenith-Equal-Area projection. This projects the sky onto a circle, and so 
# makemask just makes sure that nothing outside that circle is counted as data.

def makemask(wkernel, gassslice):
    #gassfile = '/Users/susanclark/Documents/gass_10.zea.fits'
    #gassdata  = pyfits.getdata(gassfile, 0)
    #gassslice = gassdata[45, :, :]
    
    datay, datax = np.shape(gassslice)
    
    mnvals = np.indices((datax,datay))
    pixcrd = np.zeros((datax*datay,2), np.float_)
    pixcrd[:,0] = mnvals[:,:][0].reshape(datax*datay)
    pixcrd[:,1] = mnvals[:,:][1].reshape(datax*datay)
    
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [1.125000000E3, 1.125000000E3]
    w.wcs.cdelt = np.array([-8.00000000E-2, 8.00000000E-2])
    w.wcs.crval = [0.00000000E0, -9.00000000E1]
    w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA']
    
    worldc = w.wcs_pix2world(pixcrd, 1)
    
    worldcra = worldc[:,0].reshape(datax,datay)
    worldcdec = worldc[:,1].reshape(datax,datay)
    
    gm = np.zeros(gassslice.shape)
    gm[worldcdec < 0] = 1
    
    gmconv = filters.correlate(gm, weights=wkernel)
    gg = copy.copy(gmconv)
    gg[gmconv < np.max(gmconv)] = 0
    gg[gmconv == np.max(gmconv)] = 1
    
    return gg

'''

#------------------------------ Ok with this >
#Performs a circle-cut of given radius on inkernel.
#Outkernel is 0 anywhere outside the window.    
def circ_kern(inkernel, radius):
    #These are all the possible (m,n) indices in the image space, centered on center pixel
    mnvals = np.indices((len(inkernel), len(inkernel)))
    kcntr = np.floor(len(inkernel)/2.0)
    mvals = mnvals[:,:][0] - kcntr
    nvals = mnvals[:,:][1] - kcntr

    rads = np.sqrt(nvals**2 + mvals**2)
    outkernel = copy.copy(inkernel)
    outkernel[rads > radius/2] = 0
    
    return outkernel

'''
I was playing with this in the scrap.py file and think I get it better now.
#import scrap
#print scrap.ring(20, 6, 12) 
'''
#------------------------------ < Until Here.

#Unsharp mask. Returns binary data.
def umask(data, inkernel):    
    outdata = filters.correlate(data, weights=inkernel)
    
    #I don't understand what kernweight does..
    #Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(inkernel, axis=0)
    kernweight = np.sum(kernweight, axis=0)
    subtr_data = data - outdata/kernweight
    
    #Convert to binary data
    bindata = copy.copy(subtr_data)
    bindata[subtr_data > 0] = 1
    bindata[subtr_data <= 0] = 0

    return bindata

def fast_hough(in_arr, xyt, ntheta):
    incube = np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)
    out = np.sum(np.sum(incube*xyt,axis=0), axis=0)
    
    return out        


#------------------------------ Got it >
def all_thetas(window, thetbins):
    wx, wy = window.shape #Parse x/y dimensions
    ntheta = len(thetbins) #Parse height in theta
    
    #Makes prism; output has dimensions (x, y, theta)
    out = np.zeros((wx, wy, ntheta), np.int_)
    
    for i in xrange(wx):
        for j in xrange(wy):
            #At each x/y value, create new single-pixel image
            w_1 = np.zeros((wx, wy), np.float_)
            
            # run the Hough for each point one at a time
            if window[i,j] == 1:
                w_1[i,j] = 1
                #------------------------------ < Until Here.
                H, thets, dist = houghnew(w_1, thetbins) 
                rel = H[np.floor(len(dist)/2), :]
                out[i, j, :] = rel
      
    return out    

def houghnew(img, theta=None, idl=False):
    if img.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180)
    
    wx, wy = img.shape    
    wmid = np.floor(wx/2)
    
    if idl == True:
        print 'idl values'
        #Here's that ntheta again..
        ntheta = math.ceil((np.pi*np.sqrt(2)*((wx-1)/2.0)))  
        theta = np.arange(0, np.pi, np.pi/ntheta)
        dtheta = np.pi/ntheta

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

        # round the distances to the nearest integer
        # and shift them to a nonzero bin
        shifted = np.round(distances) - bins[0]

        # cast the shifted values to ints to use as indices
        indices = shifted.astype(np.int)

        # use bin count to accumulate the coefficients
        bincount = np.bincount(indices)

        # finally assign the proper values to the out array
        out[:len(bincount), i] = bincount

    return out, theta, bins

#------------------------------ Got it >
def window_step(data, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask):    
    #Circular kernels
    wsquare1 = np.ones((wlen, wlen), np.int_) #Square of 1s
    kernel = circ_kern(wsquare1, smr) #Stores an smr-sized circle
    wkernel = circ_kern(wsquare1, wlen) #And an wlen-sized circle

	#------------------------------ < Until Here.
    xyt = all_thetas(wkernel, theta) #Not sure 
    print xyt.shape, 'xyt shape'
    
    #unsharp mask the whole data set
    udata = umask(data, kernel)
    
    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(wkernel, xyt, ntheta)
    
    start = time.clock()
    
    Hthets = []
    Hi = []
    Hj = []
    
    start0=time.clock()
    dcube = np.repeat(udata[:,:,np.newaxis], repeats=ntheta, axis=2)
    end0 = time.clock()
    print 'cube data', end0-start0
    
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    npsum = np.sum

    #Loop: (j,i) are centerpoints of data window.
    for j in xrange(datay):
        print j 
        #if j >= ucntr and j < 1125:
        if j >= ucntr and j < (datay - ucntr):
            for i in xrange(datax):
                #if i >= 1125 and i < (datax - ucntr):
                if i >= ucntr and i < (datax - ucntr):
                    #if mask[i,j] == 1: #Only necessary for GASS data
                    wcube = dcube[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1,:]   
                    
                    h = npsum(npsum(wcube*xyt,axis=0), axis=0)
                    
                    hout = h/h1 - frac
                    hout[hout<0.0] = 0.0
                
                    if np.sum(hout) > 0:
                        htapp(hout)
                        hiapp(i)
                        hjapp(j)
                    

        
    end = time.clock()
    print 'Code time %.6f seconds' % (end - start)         
    
    return Hthets, Hi, Hj

#*********************************
print 'processing galfa'

#To run the code, three things need to be called: getData, setParams, and window_step. The output is saved as follows
#(this could easily be wrapped, it just currently isn't).

#Modified the getData function and got it to run on my pc!
gassslice, datay, datax = getData('test.fits')  #was getData('null',4,6)
print 'Successfully got Data!'

wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask = setParams(gassslice, 125, 5, 0.70)
print 'Successfully set Params!'

print 'Running window_step...'
Hthets, Hi, Hj = window_step(gassslice, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask) 
hi = np.array(Hi)
hj = np.array(Hj)
hthets = np.array(Hthets)

np.save('test_hi.npy', hi)
np.save('test_hj.npy', hj)
np.save('test_hthets.npy', hthets)

