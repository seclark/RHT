#!/usr/bin/python
#FAST ROLLING HOUGH TRANSFORM
#Susan Clark, Lowell Schudel

#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------
from __future__ import division #Must be first line of code in the file
from astropy.io import fits

import scipy.ndimage
import math
import os
import sys
import string
import tempfile 
import shutil
import time 
import fnmatch

import matplotlib.pyplot as plt
import numpy as np


#-----------------------------------------------------------------------------------------
#Initialization 1 of 3: Calculation Parameters
#-----------------------------------------------------------------------------------------

WLEN = 55 #101.0 #Diameter of a 'window' to be evaluated at one time
FRAC = 0.70 #0.70 #fraction (percent) of one angle that must be 'lit up' to be counted
SMR = 15 #smoothing radius of unsharp mask function

ORIGINAL = True #Compute exactly the RHT looking along diameters #False causes it to be single-sided

#-----------------------------------------------------------------------------------------
#Initialization 2 of 3: Runtime Variable
#-----------------------------------------------------------------------------------------

#Output Formatting
OUTPUT = '.' #Directory for RHT output
xyt_format = '.fits' #'.npz' #Numpy arrays will be compressed, but fits are more standardized
xyt_suffix = '_xyt' #Arbitrary name indicating the output of the RHT accumulator array 
DIGITS = 2 #Limits ouput to 100 files (_xyt00.fits to _xyt99.fits)

#User Interface
TEXTWIDTH = 70 #Width of some displayed text objects

#Allows processing of larger files than RAM alone would allow
BUFFER = True #Gives the program permission to create a temporary directory for RHT data
DTYPE = np.float32 #Single precision
FILECAP = int(5e8) #Maximum number of BYTES allowed for a SINGLE buffer file. THERE CAN BE MULTIPLE BUFFER FILES!

#Excluded Data Types 
BAD_0 = True
BAD_INF = True
BAD_Neg = False 

#Timers
start_time = None
stop_time = None

#-----------------------------------------------------------------------------------------
#Utility Functions
#-----------------------------------------------------------------------------------------
def announcement(strings):
    result = ''
    if type(strings) == list:
        strings.append('*'*TEXTWIDTH)
        strings.insert(0, '*'*TEXTWIDTH)
        result = '\n'.join(string.center(str(s), TEXTWIDTH, ' ') for s in strings) 
    elif type(strings) == str:
        result = announcement(strings.split('\n'))
    else:
        result = announcement(str(strings))
    return result

def announce(strings):
    print announcement(strings)

def update_progress(progress, message='Progress:', final_message='Finished:'):
    #Create progress meter that looks like: 
    #message + ' ' + '[' + '#'*p + ' '*(length-p) + ']' + time_message

    if not 0.0 <= progress <= 1.0:
        raise ValueError('Progress value outside allowed value in update_progress') 

    global start_time
    global stop_time 

    if 0.0 == progress: #First call
        start_time = time.time()
        stop_time = None
        return

    elif stop_time is None: #Second call
        stop_time = start_time + (time.time() - start_time)/progress

    elif np.random.rand() > 0.98: #Randomly callable re-calibration
        stop_time = start_time + (time.time() - start_time)/progress

    #Normal Call with Progress
    sec_remaining = int(stop_time - time.time())
    if sec_remaining >= 60:
        time_message = ' < ' + str(sec_remaining//60  +1) + 'min'
    else:
        time_message = ' < ' + str(sec_remaining +1) + 'sec'

    length = int(0.55 * TEXTWIDTH)
    messlen = TEXTWIDTH-(length+3)-len(time_message)
    message = string.ljust(message, messlen)[:messlen]

    p = int(length*progress/1.0) 
    sys.stdout.write('\r{2} [{0}{1}]{3}'.format('#'*p, ' '*(length-p), message, time_message))
    sys.stdout.flush()

    if p == length: #Final Call
        total = int(time.time()-start_time)
        if total > 60:
            time_message = ' ' + str(total//60) + 'min'
        else:
            time_message = ' ' + str(total) + 'sec'
        
        final_offset = TEXTWIDTH-len(time_message)
        final_message = string.ljust(final_message, final_offset)[:final_offset]
        sys.stdout.write('\r{0}{1}'.format(final_message, time_message))
        sys.stdout.flush()
        start_time = None
        stop_time = None
        print ''

#-----------------------------------------------------------------------------------------
#Naming Conventions and Converisons
#-----------------------------------------------------------------------------------------

def filename_from_path(filepath):
    #Maintains all characters in path except for those after and including the last period
    return os.path.basename('.'.join( filepath.split('.')[ 0:filepath.count('.') ] ) ) 

def xyt_name_factory(filepath, wlen, smr, frac, original=ORIGINAL):
    #Returns the filename that _xyt output should have.
    #Will have the general behavior: filename_xyt00.format

    #filepath ~ dirname/name.fits
    #filename ~ dirname/name
    #fnmatch_string ~ name + xyt_suffix + ?? + xyt_format

    filename = filename_from_path(filepath) #Removes RHT-specific endings
    dirname = os.path.dirname(os.path.abspath(filepath))
    fnmatch_string = filename + xyt_suffix + '?'*DIGITS + xyt_format 
    xyt_files = fnmatch.filter(os.listdir(dirname), fnmatch_string) 
    xyt_array = [None]*(10**DIGITS) 

    #Tries to find a parameter match among existing files
    left = string.find(fnmatch_string, '?')
    for x in xyt_files:
        abs_x = os.path.join(dirname, x)
        if getXYT(abs_x, match_only={'WLEN':wlen, 'SMR':smr, 'FRAC':frac, 'ORIGINAL':original} ): 
            return os.path.normpath(abs_x)
        else:
            xyt_array[int( x[left:(left+DIGITS)] )] = x
    
    #Tries to find the lowest-numbered name that is unoccupied
    for i, y in enumerate(xyt_array):
        if y is None:
            #print 'Found _xyt available for these parameters!'
            int_string = string.zfill(str(i), DIGITS)[:DIGITS] 
            xyt_filename = filename+ xyt_suffix+ int_string+ xyt_format
            return os.path.normpath(os.path.join(dirname, xyt_filename))
    
    #Failure: No match and no available output slots
    xyt_filename = string.replace(fnmatch_string, '?', '0') #Filename0
    print 'In xyt_filename(): No existing ouput matches the input parameters and no namespace is available'
    print 'Overwrite ' + xyt_filename + '?..' 
    choice = raw_input(' [y]/n/'+'0'*(DIGITS-1)+'x')
    if len(choice) == 0 or choice == 'y':
        return os.path.normpath(os.path.join(dirname, xyt_filename))
    elif choice != 'n':
        int_string = string.zfill(str(int(choice)), DIGITS)[:DIGITS] 
        xyt_filename = filename+ xyt_suffix+ int_string+ xyt_format
        return os.path.normpath(os.path.join(dirname, xyt_filename)) 
    else:
        raise RuntimeError('In xyt_filename(): No existing ouput matches the input parameters and no namespace is available')

#-----------------------------------------------------------------------------------------
#Image Processing Functions
#-----------------------------------------------------------------------------------------
def is_valid_file(filepath):
    '''
    filepath: Potentially a string path to a source data

    return: Boolean, True ONLY when the data could have rht() applied successfully
    '''
    excluded_file_endings = ['_xyt.fits', '_backproj.npy', '_spectrum.npy', '_plot.png', '_result.png'] 
    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False
    
    excluded_file_content = ['_xyt', '_backproj', '_spectrum', '_plot', '_result'] 
    if any([e in filepath for e in excluded_file_content]):
        return False

    return True

def ntheta_w(w=WLEN):
    #Returns the number of theta bins in each Hthet array

    #Linearly proportional to wlen
    return int(math.ceil( np.pi*(w-1)/np.sqrt(2.0) ))  

def center(filepath, shape=(512, 512)):
    #Returns a cutout from the center of the data
    x, y = int(shape[0]), int(shape[1])
    if x < 64 or y < 64:
        raise ValueError('Invalid shape in center(), or image too small')
    data = getData(filepath)
    datay, datax = data.shape 
    if 0 < x < datax and 0 < y < datay:
        left = int(datax//2-x//2)
        right = int(datax//2+x//2)
        up = int(datay//2+y//2)
        down = int(datay//2-y//2)
        cutout = np.array(data[down:up, left:right])
        filename = filename_from_path(filepath)
        center_filename = filename+'_center.npy'
        np.save(center_filename, cutout)
        return center_filename
    else:
        return center(filepath, shape=(x//2,y//2))

def putXYT(xyt_filename, hi, hj, hthets, wlen, smr, frac, original, backproj=None, compressed=True):
    #Checks for existing _xyt arrays 
    #filename = filename_from_path(filepath)
    #existing_xyts = fnmatch.filter(os.listdir(os.path.dirname(os.path.realpath(xyt_filename))), filename+'_xyt??.*')
    #print existing_xyts

    if xyt_filename.endswith('.npz'):
        #IMPLEMENTATION1: Zipped Numpy arrays of Data
        if compressed:
            save = np.savez_compressed  
        else:
            save = np.savez
        if backproj is None:
            save(xyt_filename, hi=hi, hj=hj, hthets=hthets, wlen=wlen, smr=smr, frac=frac, original=original, ntheta=hthets.shape[1])
        else:
            save(xyt_filename, hi=hi, hj=hj, hthets=hthets, wlen=wlen, smr=smr, frac=frac, original=original, ntheta=hthets.shape[1], backproj=backproj)


    elif xyt_filename.endswith('.fits'):
        #IMPLEMENTATION2: FITS Table File
        Hi = fits.Column(name='hi', format='1I', array=hi)
        Hj = fits.Column(name='hj', format='1I', array=hj)
        ntheta = hthets.shape[1]
        Hthets = fits.Column(name='hthets', format=str(int(ntheta))+'E', array=hthets)
        cols = fits.ColDefs([Hi, Hj, Hthets])
        tbhdu = fits.BinTableHDU.from_columns(cols)

        #Header Values for RHT Parameters
        prihdr = fits.Header()
        prihdr['WLEN'] = wlen 
        prihdr['SMR'] = smr
        prihdr['FRAC'] = frac
        prihdr['ORIGINAL'] = ORIGINAL

        #Other Header Values
        prihdr['NTHETA'] = ntheta

        #Whole FITS File
        prihdu = fits.PrimaryHDU(data=backproj, header=prihdr)
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(xyt_filename, output_verify='silentfix', clobber=True, checksum=True)


    else:
        raise ValueError('Supported output types in putXYT include .npz and .fits only')

def getXYT(xyt_filename, match_only=False, rebuild=False, filepath=None):    
    #Reads in a .npz file containing coordinate pairs in data space (hi, hj)
    #And Hough space arrays covering theta space at each of those points
    if not os.path.isfile(xyt_filename):
        if match_only:
            return False
        else:
            raise ValueError('Input xyt_filename in getXYT matches no existing file')
    else:
        if xyt_filename.endswith('.npz'):
            data = np.load(xyt_filename, mmap_mode='r') #Allows for reading in very large files!
            Hi = data['hi']
            Hj = data['hj']
            Hthets = data['hthets']
            if match_only:
                try:
                    return all([ match_only[x] == data[string.lower(x)] for x in match_only.keys() ])
                except KeyError:
                    return False

        elif xyt_filename.endswith('.fits'):
            hdu_list = fits.open(xyt_filename, mode='readonly', memmap=True, save_backup=False, checksum=True)
            header = hdu_list[0].header
            if match_only:
                try:
                    return all([ match_only[x] == header[string.upper(x)] for x in match_only.keys() ])
                except KeyError:
                    return False

            data = hdu_list[1].data
            Hi = data['hi'] 
            Hj = data['hj'] 
            Hthets = data['hthets']

        else:
            raise ValueError('Supported input types in getXYT include .npz and .fits only')

    #Formats output properly
    if rebuild and filepath is not None:
        #Can recreate an entire 3D array of mostly 0s
        data = getData(filepath)
        datay, datax = data.shape
        ntheta = Hthets[0].shape  
        if BUFFER:
            xyt = np.memmap(tempfile.TemporaryFile(), dtype=DTYPE, mode='w+', shape=(datay, datax, ntheta))
            xyt.fill(0.0)
        else:
            print 'Warning: Reconstructing very large array in memory! Set BUFFER to True!' 
            xyt = np.zeros((datay, datax, ntheta))
        coords = zip(Hj, Hi)
        for c in range(len(coords)):
            j,i = coords[c]
            xyt[j,i,:] = Hthets[c]
        return xyt
    else:
        #Returns the sparse, memory mapped form only
        return Hi, Hj, Hthets   


def bad_pixels(data):
    #Returns an array of the same shape as data
    #NaN values MUST ALWAYS be considered bad.
    #Bad values become 1, all else become 0
    data = np.array(data, np.float)
    
    #IMPLEMENTATION1: Do Comparisons which are VERY different depending on boolean choices 
    try:
        if BAD_INF:
            if BAD_0:
                if BAD_Neg:
                    return np.logical_or(np.logical_not(np.isfinite(data)), np.less_equal(data, 0.0))
                else:    
                    return np.logical_or(np.logical_not(np.isfinite(data)), np.equal(data, 0.0))
            else:
                if BAD_Neg:
                    return np.logical_or(np.logical_not(np.isfinite(data)), np.less(data, 0.0))
                else:    
                    return np.logical_not(np.isfinite(data))
        else:
            if BAD_0:
                if BAD_Neg:
                    return np.logical_or(np.isnan(data), np.less_equal(data, 0.0))
                else:    
                    return np.logical_not(np.nan_to_num(data)) #(Nans or 0) ---> (0) ---> (1)
            else:
                if BAD_Neg:
                    return np.logical_or(np.isnan(data), np.less(data, 0.0))
                else:    
                    return np.isnan(data)
        '''
        #IMPLEMENTATION2: Map values determined by flags into the data array
        not_that = np.zeros_like(data)
        infs = np.empty_like(data).fill(BAD_INF)
        zeros = np.empty_like(data).fill(BAD_0)
        negs = np.empty_like(data).fill(BAD_Neg)
        
        isinf = np.where(np.isinf(data), infs, not_that)
        iszero = np.where(np.logical_not(data), zeros, not_that)
        isneg = np.where(np.less(0.0), negs, not_that)
        return np.logical_or(np.isnan(data), np.logical_or(isinf, np.logical_or(iszero, isneg)))
        '''
    except:
        #IMPLEMENTATION3: Give up?
        print 'Unable to properly mask data in bad_pixels()...'
        return data.astype(np.bool)

def all_within_diameter_are_good(data, diameter):
    assert diameter%2
    r = int(np.floor(diameter/2))

    #Base case, 'assume all pixels are bad'
    mask = np.zeros_like(data)

    #Edge case, 'any pixel not within r of the edge might be ok'
    datay, datax = data.shape
    mask[r:datay-r, r:datax-r] = 1

    #Identifiably bad case, 'all pixels within r of me are not bad'
    circle = circ_kern(diameter)
    y_arr, x_arr = np.nonzero(circle)
    y_arr = y_arr - r
    x_arr = x_arr - r

    #IMPLEMENTATION1: Zero any mask pixel within r of a bad pixel
    update_progress(0.0)
    coords = zip(*np.nonzero(bad_pixels(data)))
    N = len(coords)
    for c in range(N):    
        j,i = coords[c]
        x = (x_arr + i).astype(np.int).clip(0, datax-1)
        y = (y_arr + j).astype(np.int).clip(0, datay-1)
        mask[y, x] = 0 
        update_progress((c+1)/float(N), message='Masking:', final_message='Finished Masking:') 
    '''
    #IMPLEMENTATION2: For each good pixel, 'Not Any Bad pixels near me'
    coords = zip(*np.nonzero(mask))
    for c in range(len(coords)):
        j,i = coords[c]
        x = (x_arr + i).astype(np.int).clip(0, datax-1)
        y = (y_arr + j).astype(np.int).clip(0, datay-1)
        mask[j][i] = np.logical_not(np.any(bad_pixels( data[y, x] )))
    '''
    return mask 

def getData(filepath, make_mask=False, smr=SMR, wlen=WLEN):
    #Reads in and makes proper masks for images from various sources
    #smr_mask masks any pixel within smr of any bad pixels, and the edge
    #wlen_mask masks any pixel within wlen of any bad pixels, and the edge
    #Supports .fits, .npy, and PIL formats
    
    try:
        #Reading Data
        if filepath.endswith('.fits'):
            #Fits file handling
            hdu = fits.open(filepath, memmap=True)[0] #Opens first HDU
            data = hdu.data #Reads all data as an array

        elif filepath.endswith('.npy'):
            data = np.load(filepath, mmap_mode='r') #Reads numpy files 
        
        else:
            data = scipy.ndimage.imread(filepath, flatten=True)[::-1] #Makes B/W array, reversing y-coords         
    except:
        #Failure Reading Data
        if make_mask:
            print 'Failure in getData('+filepath+')... Returning None, None, None'
            return None, None, None
        else:
            print 'Failure in getData('+filepath+')... Returning None'
            return None 

    if not make_mask:
        #Done Reading Data, No Mask Needed
        return data
    else:
        #Mask Needed, cuts away smr radius from bads, then wlen from bads 
        smr_mask = all_within_diameter_are_good(data, 2*smr+1)
        nans = np.empty(data.shape, dtype=np.int).fill(np.nan)
        wlen_mask = all_within_diameter_are_good( np.where(smr_mask, data, nans), wlen)

        '''
        datamin = np.nanmin(data)
        datamax = np.nanmax(data)
        datamean = np.nanmean(data)
        print 'Data File Info, Min:', datamin, 'Mean:', datamean, 'Max:', datamax
        '''
        return data, smr_mask, wlen_mask

#Performs a circle-cut of given diameter on inkernel.
#Outkernel is 0 anywhere outside the window.   
def circ_kern(diameter):
    assert diameter%2
    r = diameter//2 #int(np.floor(diameter/2))
    mnvals = np.indices((diameter, diameter)) - r
    rads = np.hypot(mnvals[0], mnvals[1])
    return np.less_equal(rads, r).astype(np.int)

#Unsharp mask. Returns binary data.
def umask(data, radius, smr_mask=None):
    assert data.ndim == 2

    kernel = circ_kern(2*radius+1)
    outdata = scipy.ndimage.filters.correlate(data, kernel) #http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.filters.convolve.html#scipy.ndimage.filters.convolve
    
    #Correlation is the same as convolution here because kernel is symmetric 
    #Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(kernel)
    subtr_data = data - outdata/kernweight
    
    #Convert to binary data
    bindata = np.greater(subtr_data, 0.0)
    if smr_mask == None:
        return bindata
    else:
        return np.logical_and(smr_mask, bindata) #np.where(smr_mask, bindata, smr_mask)

def fast_hough(in_arr, xyt): #, hout=None): 

    assert in_arr.ndim == 2 
    assert xyt.ndim == 3
    assert in_arr.shape[0] == xyt.shape[0]
    assert in_arr.shape[1] == xyt.shape[1]

    #IMPLEMENTATION0: Let python figure out the implementation. (FASTEST)
    return np.einsum('ijk,ij', xyt, in_arr)
    
    '''
    if hout == None:
        return np.einsum('ijk,ij', xyt, in_arr) #, dtype=np.int) 
    else:
        assert hout.ndim == 1
        assert hout.shape[0] == xyt.shape[2]
        np.einsum('ijk,ij', xyt, in_arr, out=hout)
    '''
    #IMPLEMENTATION1: Copy 2D array into 3D stack, and multiply by other stack (SLOW)
    #cube = np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)*xyt
     
    #IMPLEMENTATION2: Broadcast 2D array against 3D stack and multiply (FAST)
    #cube = np.multiply( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt).astype(np.float, copy=False)

    #IMPLEMENTATION3: Broadcast 2D array against 3D stack and AND them together (VERY FAST)
    #assert in_arr.dtype == np.bool_ 
    #assert xyt.dtype == np.bool_  
    #cube = np.logical_and( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt)
        
    #return np.sum(np.sum( cube , axis=0, dtype=np.int), axis=0, dtype=np.float) #WORKS FAST AND DIVIDES PROPERLY
    #return np.sum(cube, axis=(1,0), dtype=np.int)

'''
try:
    from cython_hough import fast_hough as new_fast_hough 
    print 'Fast Hough Compiled in C'
    fast_hough = new_fast_hough
except:
    pass
'''
    
'''
try:
    ###Works, but without performance increase
    from numba import autojit
    fast_hough = autojit(fast_hough)

    ###
    #import numba
    #from numba.types import i1, f4
    #fast_hough = numba.jit(signature_or_function=fast_hough, argtypes=['f4[:](i1[:,:], i1[:,:,:], i1)'], target='cpu')(fast_hough)
    print 'Autojit is go!'
except:
    print 'Failure to autojit..'
    #raise
'''

def houghnew(image, cos_theta, sin_theta):
    assert image.ndim == 2 
    assert cos_theta.ndim == 1
    assert sin_theta.ndim == 1
    assert len(cos_theta) == len(sin_theta)
    assert image.shape[0] == image.shape[1]

    wmid = image.shape[0]//2 #Really means wlen//2

    # compute the vertical bins (the distances)
    nr_bins = np.ceil(np.hypot(*image.shape))

    # allocate the output data
    out = np.zeros((int(nr_bins), len(cos_theta)), dtype=np.int)

    # find the indices of the non-zero values in
    # the input data
    y, x = np.nonzero(image)

    # x and y can be large, so we can't just broadcast to 2D
    # arrays as we may run out of memory. Instead we process
    # one vertical slice at a time.
    for i, (cT, sT) in enumerate(zip(cos_theta, sin_theta)):

        # compute the base distances
        distances = (x - wmid) * cT + (y - wmid) * sT

        # round the distances to the nearest integer
        # and shift them to a nonzero bin
        shifted = np.round(distances) + nr_bins/2

        # cast the shifted values to ints to use as indices
        indices = shifted.astype(np.int)
        
        # use bin count to accumulate the coefficients
        bincount = np.bincount(indices) 

        # finally assign the proper values to the out array
        out[:len(bincount), i] = bincount
        #out.T[i] = bincount

    return out[np.floor(nr_bins/2), :]


def all_thetas(wlen, thetbins):
    assert thetbins.ndim == 1 
    assert wlen%2

    #Initialize a circular window of ones
    window = circ_kern(wlen)
    assert window.shape[0] == window.shape[1]
    if not ORIGINAL:
        window[:,:wlen//2] = 0
    
    # precompute the sin and cos of the angles
    cos_theta = np.cos(thetbins)
    sin_theta = np.sin(thetbins)
    
    #Makes prism; output has dimensions (y, x, theta)
    ntheta = len(thetbins)
    #outshape = (wlen, wlen, ntheta)
    out = np.zeros(window.shape+(ntheta,), np.int)
    coords = zip( *np.nonzero(window))
    for (j, i) in coords:
        #At each x/y value, create new single-pixel data
        w_1 = np.zeros_like(window)
        w_1[j,i] = 1
        out[j, i, :] = houghnew(w_1, cos_theta, sin_theta)

    if not ORIGINAL:
        out[:,:,ntheta//2:] = out[::-1,::-1,ntheta//2:] 
        out[:wlen//2+1,:,ntheta//2] = 0
        out[wlen//2:,:,0] = 0

    '''
    plt.imshow(out[:, :, 0])
    plt.show()
    plt.imshow(out[:, :, 1])
    plt.show()
    '''
    return out 


def theta_rht(theta_array, uv=False):
    #Maps an XYT cube into a 2D Array of angles- weighted by their significance
    if ORIGINAL:
        thetas = np.linspace(0.0, np.pi, len(theta_array), endpoint=False, retstep=False)
        ys = theta_array*np.sin(2.0*thetas)
        xs = theta_array*np.cos(2.0*thetas)    
        rough_angle = 0.5*np.arctan2(np.sum(ys), np.sum(xs)) #EQUATION (7)
        angle = np.pi-math.fmod(rough_angle+np.pi, np.pi) #EQUATION (8)
    else:
        thetas = np.linspace(0.0, 2*np.pi, len(theta_array), endpoint=False, retstep=False)
        ys = theta_array*np.sin(thetas)
        xs = theta_array*np.cos(thetas)    
        angle = np.arctan2(np.sum(ys), np.sum(xs)) 
        
    if not uv:
        return angle
    else:
        #OR It can map all arrays to the vector (x, y) of theta power at one point
        #MADE FOR USE WITH plt.quiver()
        return angle, np.cos(angle), np.sin(angle)
    
    

def buffershape(ntheta, filesize=FILECAP):
    #Shape of maximum sized array that can fit into a SINGLE buffer file 
    ntheta = int(ntheta)
    filesize = int(filesize)
    if not 0 < filesize <= FILECAP:
        print 'Chosen buffer size exceeds existing limit. Reset to', str(FILECAP), 'Bytes'
        filesize = FILECAP

    bits_per_element_in_bits = np.dtype(DTYPE).itemsize
    bits_per_file_in_bits = filesize*8
    elements_per_file_in_elements = int(bits_per_file_in_bits // bits_per_element_in_bits)
    length_in_elements = int(elements_per_file_in_elements // ntheta)
    if length_in_elements <= 0:
        print 'In buffershape, ntheta has forced your buffer size to become larger than', filesize, 'Bytes'
        length_in_elements = 1

    return (length_in_elements, ntheta) 

def concat_along_axis_0(memmap_list):
    #Combines memmap objects of the same shape, except along axis 0,
    #BY LEAVING THEM ALL ON DISK AN APPENDING THEM SEQUENTIALLY
    if len(memmap_list) == 0:
        raise ValueError('Failed to buffer any data!') 

    elif len(memmap_list) == 1:
        return memmap_list[0]
    
    else:
        '''
        #IMPLEMENTATION1: Make a new large memmapped file and sequentially dump data in
        lengths = [memmap.shape[0] for memmap in memmap_list]
        shapes = [memmap.shape[1:] for memmap in memmap_list]
        assert all([x==shapes[0] for x in shapes[1:]])

        big_memmap = np.memmap(os.path.join(temp_dir, +'rht.dat'), dtype=DTYPE, mode='r+', shape=(sum(lengths), *shapes[0])  )   
        lengths.insert(0, sum(lengths))
        for i in range(len(memmap_list)):
            temp_file = memmap_list[i]
            big_memmap[ sum(lengths[0:i]) : sum(lengths[0:i+1])-1, ...] = tempfile[:, ...]
            temp_file.flush()         
            temp_file.close()        
            del temp_file 
        return big_memmap
        '''
        #IMPLEMENTATION2: Append data to first given memmaped file, then delete and repeat
        seed = memmap_list[0]
        others = memmap_list[1:]
        lengths = [memmap.shape[0] for memmap in others]
        shapes = [memmap.shape[1:] for memmap in others]
        assert all([x==seed.shape[1:] for x in shapes])
        
        bits_per_element_in_bits = np.dtype(DTYPE).itemsize
        elements_per_shape_in_elements = np.multiply.reduce(seed.shape[1:])
        bytes_per_shape_in_bytes = elements_per_shape_in_elements * bits_per_element_in_bits // 8

        def append_memmaps(a, b):
            a.flush()
            a.close()
            c = np.memmap(a.filename, dtype=DTYPE, mode='r+', offset=bytes_per_shape_in_bytes*a.shape[0], shape=b.shape )
            c[:,...] = b[:,...] #Depends on offset correctly allocating new space at end of file
            b.flush()         
            b.close()        
            del b
            return c 

        return reduce(append_memmaps, others, initializer=seed)

def window_step(data, wlen, frac, smr, smr_mask, wlen_mask, xyt_filename, message): 

    assert frac == float(frac) #Float
    assert 0 <= frac <= 1 #Fractional
    
    assert wlen == int(wlen) #Integer
    assert wlen > 0 #Positive
    assert wlen%2 #Odd

    assert smr == int(smr) #Integer
    assert smr > 0 #Positive
    
    #Needed values
    r = wlen//2 
    ntheta = ntheta_w(wlen)
    if ORIGINAL:
        theta, dtheta = np.linspace(0.0, np.pi, ntheta, endpoint=False, retstep=True)        
    else:
        theta, dtheta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False, retstep=True)

    #Cylinder of all lit pixels along a theta value
    xyt = all_thetas(wlen, theta) 
    xyt.setflags(write=0)
    
    #Unsharp masks the whole data set
    masked_udata = umask(data, smr, smr_mask=smr_mask)
    masked_udata.setflags(write=0)

    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(circ_kern(wlen), xyt)
    h1.setflags(write=0)

    #Local function calls are faster than globals
    Hthets = []
    Hi = []
    Hj = []
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    nptruediv = np.true_divide
    npge = np.greater_equal
    
    #Bonus Backprojection Creation
    backproj = np.zeros_like(data)
    
    if not BUFFER:
        #Number of RHT operations that will be performed, and their coordinates
        coords = zip( *np.nonzero( wlen_mask))
        update_progress(0.0)
        N = len(coords)
        for c in range(N):
            j,i = coords[c]
            h = fast_hough(masked_udata[j-r:j+r+1, i-r:i+r+1], xyt)
            hout = nptruediv(h, h1)
            hout *= npge(hout, frac)
            if np.any(hout):
                htapp(hout)
                hiapp(i)
                hjapp(j)
                backproj[j][i] = np.sum(hout) 
            update_progress((c+1)/float(N), message=message, final_message=message)
            #End
        putXYT(xyt_filename, np.array(Hi), np.array(Hj), np.array(Hthets), wlen, smr, frac, backproj=np.divide(backproj, np.amax(backproj)) ) #Saves data
        return True 

    else:
        #Preparing to write hout to file during operation so it does not over-fill RAM
        temp_dir = tempfile.mkdtemp()
        temp_files = [] #List of memmap objects
        buffer_shape = buffershape(ntheta) #variable_length, ntheta
        def next_temp_filename():
            return os.path.join(temp_dir, 'rht'+ str(len(temp_files)) + '.dat')
        #print 'Temporary files in:', temp_dir 

        #Number of RHT operations that will be performed, and their coordinates
        update_progress(0.0)
        coords = zip( *np.nonzero( wlen_mask))
        N = len(coords)
        for c in range(N):
            j,i = coords[c]
            h = fast_hough(masked_udata[j-r:j+r+1, i-r:i+r+1], xyt)
            hout = nptruediv(h, h1) 
            hout *= npge(hout, frac)
            if np.any(hout):
                htapp(hout)
                hiapp(i)
                hjapp(j)
                backproj[j][i] = np.sum(hout) 
                if len(Hthets) == buffer_shape[0]:
                    temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=buffer_shape )) #Creates full memmap object
                    theta_array = np.array(Hthets, dtype=DTYPE) #Convert list to array
                    temp_files[-1][:] = theta_array[:] #Write array to last memmapped object in list
                    Hthets = [] #Reset Hthets
            update_progress((c+1)/float(N), message=message, final_message=message)
            #End

        if len(Hthets) > 0:
            temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=(len(Hthets), ntheta)  )) #Creates small memmap object
            theta_array = np.array(Hthets, dtype=DTYPE) #Convert list to array
            temp_files[-1][:] = theta_array[:] #Write array to last memmapped object in list

        #print 'Converting list of buffered hthet arrays into final XYT cube'
        converted_hthets = concat_along_axis_0(temp_files) #Combines memmap objects sequentially
        converted_hthets.flush()
        putXYT(xyt_filename, np.array(Hi), np.array(Hj), converted_hthets, wlen, smr, frac, backproj=np.divide(backproj, np.amax(backproj)) ) #Saves data
        
        del converted_hthets

        
        def rmtree_failue(function, path, excinfo):
            try:
                #os.listdir(temp_dir):
                for obj in temp_files: 
                    #q = file(obj)
                    #q.close()
                    #del q
                    obj.close()
                    os.remove(obj)
                os.removedirs(temp_dir)
            except:
                print 'Failed to delete:', path 

        shutil.rmtree(temp_dir, ignore_errors=False, onerror=rmtree_failue)

        return True

#-----------------------------------------------------------------------------------------
#Interactive Functions
#-----------------------------------------------------------------------------------------

def rht(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR, original=ORIGINAL):
    '''
    filepath: String path to source data, which will have the Rolling Hough Transform applied
    force: Boolean indicating if rht() should still be run, even when output exists for these inputs

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an data

    Saves:
        X-Y-ThetaPower Array --> name_xyt.npz

    return: Boolean, if the function succeeded
    '''

    assert frac == float(frac) #Float
    assert 0 <= frac <= 1 #Fractional
    
    assert wlen == int(wlen) #Integer
    assert wlen > 0 #Positive
    assert wlen%2 #Odd

    assert smr == int(smr) #Integer
    assert smr > 0 #Positive

    if not is_valid_file(filepath):
        #Checks to see if a file should have the rht applied to it...
        print 'Invalid filepath encountered in rht('+filepath+')...'
        return False

    try:
        xyt_filename = xyt_name_factory(filepath, wlen, smr, frac, original=ORIGINAL)
        
        if not force and os.path.isfile(xyt_filename):
            return True

        print '1/4:: Retrieving Data from:', filepath
        data, smr_mask, wlen_mask = getData(filepath, make_mask=True, smr=smr, wlen=wlen)
        datay, datax = data.shape

        print '2/4::', 'Size:', str(datax)+'x'+str(datay)+',', 'Wlen:', str(wlen)+',', 'Smr:', str(smr)+',', 'Frac:', str(frac)
        
        message = '3/4:: Running RHT...'
        success = window_step(data, wlen, frac, smr, smr_mask, wlen_mask, xyt_filename, message) 
        

        print '4/4:: Successfully Saved Data As', xyt_filename
        return success
    
    except:
        raise #__________________________________________________________________________________________________________ Raise
        return False


def interpret(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR, original=ORIGINAL):
    '''
    filepath: String path to source data, used in forcing and backprojection
    force: Boolean indicating if rht() should be run, even when required_files are found

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an data

    Saves:
        Backprojection --> name_backproj.npy
        ThetaSpectrum --> name_spectrum.npy

    return: Boolean, if the function succeeded
    '''
    #Makes sure relevant files are present! 
    filename = filename_from_path(filepath)
    xyt_filename = filename + '_xyt.fits'
    required_files = [xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing or force:
        #Runs rht(filepath), after clearing old output, since that needs to be done
        for f in required_files:
            try:
                #Try deleting obsolete output
                os.remove(f)
            except:
                #Assume it's not there
                continue 
        rht(filepath, force=force, wlen=wlen, frac=frac, smr=smr, original=ORIGINAL) 

    #Proceed with iterpreting the rht output files
    hi, hj, hthets = getXYT(xyt_filename, rebuild=False)

    #Spectrum *Length ntheta array of theta power (above the threshold) for whole data*
    '''
    if ORIGINAL:
        theta, dtheta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False, retstep=True)
        specturm = 
    else:
    '''
    spectrum = np.sum(hthets, axis=0)
    spectrum_filename = filename + '_spectrum.npy'
    np.save(spectrum_filename, np.array(spectrum))

    #Backprojection only *Full Size* #Requires data
    data = getData(filepath)
    datay, datax = data.shape
    backproj = np.zeros_like(data)
    coords = zip(hi, hj)
    for c in range(len(coords)):
        backproj[coords[c][1]][coords[c][0]] = np.sum(hthets[c]) 
    np.divide(backproj, np.sum(backproj), backproj)
    backproj_filename = filename + '_backproj.npy'
    np.save(backproj_filename, np.array(backproj))

    print 'Interpreting... Success'
    return True

    
def viewer(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR, original=ORIGINAL):
    '''
    filepath: String path to source data, used in forcing and backprojection
    force: Boolean indicating if interpret() should be run, even when required_files are found

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an data
    
    Plots:
        name_backproj.npy --> Backprojection 
        name_spectrum.npy --> ThetaSpectrum, Linearity

    return: Boolean, if the function succeeded
    '''
    #Makes sure relevant files are present!____________________
    filename = filename_from_path(filepath)
    xyt_filename = filename + '_xyt.fits'  
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'
    required_files = [backproj_filename, spectrum_filename, xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing or force:
        #Interprets file, after clearing old output, since that needs to be done
        for f in required_files:
            try:
                #Try deleting obsolete output
                os.remove(f)
            except:
                #Assume it's not there
                continue 
        interpret(filepath, force=force, wlen=wlen, frac=frac, smr=smr, original=ORIGINAL) 

    #Loads in relevant files and data
    data = getData(filepath) #data, smr_mask, wlen_mask = getData(filepath, make_mask=True, smr=smr, wlen=wlen)
    backproj = np.load(backproj_filename).astype(np.float)
    spectrum = np.load(spectrum_filename).astype(np.float)
    hi, hj, hthets = getXYT(xyt_filename)

    #Gather parameters from that data
    ntheta = len(spectrum)
    datay, datax = data.shape

    #Produce Specific Plots
    masked_udata = umask(data, smr)
    log = np.log(np.where( np.isfinite(data), data, np.ones_like( data) ))
    U = np.zeros_like(hi)
    V = np.zeros_like(hj)
    C = np.zeros((len(U)), dtype=np.float)
    coords = zip(hi, hj)
    for c in range(len(coords)):
        C[c], U[c], V[c] = theta_rht(hthets[c], uv=True)
    C *= np.isfinite(C)
    if ORIGINAL:
        C /= np.pi
    else:
        C /= 2*np.pi 
    np.clip(C, 0.0, 1.0, out=C)

    #Define Convenience Functions
    def cleanup():
        plt.clf()
        plt.cla()
        plt.close()

    #---------------------------------------------------------------------- PLOT 1
    print 'Plotting Whole Figure...'
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    #### Log-scale plot of original image data
    axes[0][0].imshow(log, cmap='gray')
    axes[0][0].set_ylim([0, datay])
    axes[0][0].set_title(filepath)

    #### Backprojection of RHT data
    axes[1][1].imshow(backproj, cmap='gray') #cmap='binary')
    axes[1][1].set_ylim([0, datay])
    axes[1][1].set_title(backproj_filename)
    
    #### Masked Image
    axes[0][1].imshow(masked_udata, cmap='gray') #cmap='binary')
    axes[0][1].set_ylim([0, datay])
    axes[0][1].set_title('Sharpened')

    #### Quiver plot of theta<RHT> across the image
    reduction = np.nonzero(C)[::1]
    axes[1][0].quiver(hi[reduction], hj[reduction], U[reduction], V[reduction], C[reduction], units='xy', pivot='middle', scale=2, cmap='hsv')
    axes[1][0].set_ylim([0, datay])
    axes[1][0].set_title('Theta<RHT>')

    #### Save and Clean up
    plt.savefig(filename + '_plot.png')
    plt.show(fig)
    cleanup()

    #---------------------------------------------------------------------- PLOT 2
    print 'Backprojecting...'

    #### Log-scale plot of original image data
    plt.subplot(121)
    plt.imshow(log, cmap='gray')
    plt.ylim(0, datay)
    plt.title(filepath)

    #### Backprojection of RHT data
    plt.subplot(122)
    plt.contour(backproj) #plt.imshow(backproj, cmap='binary')
    plt.ylim(0, datay)
    plt.title(backproj_filename)
    
    #### Save and Clean up
    plt.savefig(filename + '_result.png')
    plt.show()
    cleanup()
    
    '''
    #---------------------------------------------------------------------- PLOT 3
    print 'Theta Spectrum'

    plt.plot(np.linspace(0.0, 2*180.0, num=ntheta, endpoint=False), spectrum) #___________---2*
    
    plt.show()
    cleanup()
    
    '''
    #---------------------------------------------------------------------- PLOT 4
    #Polar plot of theta power
    print 'Linearity'
    
    if ORIGINAL:
        modified_spectrum = np.true_divide(np.append(spectrum, spectrum), 2.0) 
        plt.polar(np.linspace(0.0, 2*np.pi, num=2*ntheta, endpoint=False), modified_spectrum)
    else:
        plt.polar(np.linspace(0.0, 2*np.pi, num=ntheta, endpoint=False), spectrum)
    plt.show()
    cleanup()
    

    #---------------------------------------------------------------------- PLOTS END
    #Clear all and exit successfully
    cleanup()
    return True


def main(source=None, display=False, force=False, wlen=WLEN, frac=FRAC, smr=SMR, original=ORIGINAL):
    '''
    source: A filename, or the name of a directory containing files to transform
    display: Boolean flag determining if the input is to be interpreted and displayed
    force: Boolean flag determining if rht() will be run, when output already exists

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an data

    return: Boolean, if the function succeeded
    '''
    #Ensure that the input is a non-None, non-Empty string
    while source is None or type(source) != str or len(source)==0:
        try:
            source = raw_input('Source:')
        except:
            source = None
    
    #Interpret whether the Input is a file or directory, excluding all else
    pathlist = []
    if os.path.isfile(source):
        #Input = File 
        pathlist.append(source)
    elif os.path.isdir(source):
        #Input = Directory 
        for obj in os.listdir(source):
            obj_path = os.path.join(source, obj)
            if os.path.isfile(obj_path):
                pathlist.append(obj_path)
    else:
        #Input = Neither   
        print 'Invalid source encountered in main(); must be file or directory.'
        return False

    pathlist = filter(is_valid_file, pathlist)
    if len(pathlist) == 0:
        print 'Invalid source encountered in main(); no valid images found.'
        return False
    #____________________________________________________________________________SOURCE IS CLEAN

    #Run RHT Over All Valid Inputs 
    announce(['Fast Rolling Hough Transform by Susan Clark', 'Started for: '+source])

    summary = []
    for path in pathlist:
        success = True
        try:
            if (display):
                success = viewer(path, force=force, wlen=wlen, frac=frac, smr=smr, original=ORIGINAL)
            else:
                success = rht(path, force=force, wlen=wlen, frac=frac, smr=smr, original=ORIGINAL)
        except:
            success = False
            raise #____________________________________________________________________________________________________________________HANG??
        finally:
            if success:
                summary.append(path+': Passed')
            else:
                summary.append(path+': Failed')
    summary.append('Complete!')
    announce(summary)
    return True


#-----------------------------------------------------------------------------------------
#Initialization 3 of 3: Precomputed Objects
#-----------------------------------------------------------------------------------------


        
#-----------------------------------------------------------------------------------------
#Command Line Mode
#-----------------------------------------------------------------------------------------

if __name__ == "__main__":
    help = '''Rolling Hough Transform Utility

Command Line Argument Format:
 >>>python rht.py arg1 arg2 ... argn 

NO ARGS:
 Displays README and exits
 >>>python rht.py

SINGLE ARGS:
 pathname ==> Input file or directory to run the RHT on
 >>>python rht.py dirname/filename.fits
  
 -h ==> Displays this message
 >>>python rht.py help

 -p ==> Displays Default Params
 >>>python rht.py -p
 
MULTIPLE ARGS:
 Creates 'dirname/filename_xyt.npz' for each input data
 1st ==> Path to input file or directory
 2nd:nth ==> Named inputs controlling params and flags

  Flags: 
  -d  #Ouput is to be Displayed
  -f  #Exisitng _xyt.fits is to be Forcefully overwritten

  Params:
  -wlen=value  #Sets window diameter
  -smr=value  #Sets smoothing radius
  -frac=value  #Sets theta power threshold'''
    
    if len(sys.argv) == 1:
        #Displays the README file   
        README = 'README'
        try:
            readme = open(README, 'r')
            print readme.read(2000) 
            if len(readme.read(1)) == 1:
                print ''
                print '...see', README, 'for more information...'
                print ''
            readme.close()
        except:
            announce(help)

    elif len(sys.argv) == 2:
        #Parses input for single argument flags
        SOURCE = sys.argv[1]
        if SOURCE.lower() in ['help', '-help', 'h', '-h']:
            announce(help)
        elif SOURCE.lower() in ['params', 'param', 'p', '-p', '-params', '-param']:
            params = ['Default RHT Parameters:']
            params.append('wlen = '+str(WLEN))
            params.append('smr = '+str(SMR))
            params.append('frac = '+str(FRAC))
            announce(params)
        else:
            main(source=SOURCE)

    else:
        SOURCE = sys.argv[1]
        args = sys.argv[2:]
        
        #Default flag values
        DISPLAY = False
        FORCE = False
        
        #Default param values
        wlen = WLEN
        frac = FRAC
        smr = SMR
        original = ORIGINAL

        for arg in args:
            if '=' not in arg:
                #FLAGS which DO NOT carry values 
                if arg.lower() in ['d', '-d', 'display', '-display' ]:
                    DISPLAY = True
                elif arg.lower() in ['f', '-f', 'force', '-force' ]:
                    FORCE = True
                else:
                    print 'UNKNOWN FLAG:', arg
            else:
                #PARAMETERS which DO carry values
                argname = arg.lower().split('=')[0]
                argval = arg.lower().split('=')[1] 
                if argname in ['w', 'wlen', '-w', '-wlen']:
                    wlen = float(argval)
                elif argname in ['s', 'smr', '-s', '-smr']:
                    smr = float(argval)
                elif argname in ['f', 'frac', '-f', '-frac']:
                    frac = float(argval)
                elif argname in ['o', 'original', '-o', '-original', 'orig', '-orig']:
                    original = argval
                else:
                    print 'UNKNOWN PARAMETER:', arg

        main(source=SOURCE, display=DISPLAY, force=FORCE, wlen=wlen, frac=frac, smr=smr, original=original)

    exit()

#-----------------------------------------------------------------------------------------
#Attribution
#-----------------------------------------------------------------------------------------

#This is the Rolling Hough Transform, described in Clark, Peek, Putman 2014 (arXiv:1312.1338).
#Modifications to the RHT have been made by Lowell Schudel, CC'16.
