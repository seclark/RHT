#!/usr/bin/env python3
#FAST ROLLING HOUGH TRANSFORM
#Susan Clark, Lowell Schudel

#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------
from __future__ import division #Must be first line of code in the file
from __future__ import print_function
from builtins import filter, input, zip, range
from astropy.io import fits
import argparse
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

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
# Initialization 1 of 3: Calculation Parameters
#-----------------------------------------------------------------------------------------
 
# Diameter of a 'window' to be evaluated at one time
WLEN = 55

# Fraction (percent) of one angle that must be 'lit up' to be counted
FRAC = 0.70

# Smoothing radius of unsharp mask function 
SMR = 15 

# Compute the standard RHT (False sets the dRHT)
ORIGINAL = True 

#-----------------------------------------------------------------------------------------
# Initialization 2 of 3: Runtime Variable
#-----------------------------------------------------------------------------------------

# Optional Local Files

# Name of Readme file included with this software
README = 'README' 

# Output Formatting
# Directory for RHT output
OUTPUT = '.'
if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)

# Output format is standard fits. In the future we may support saved numpy arrays.
xyt_format = '.fits'

# String that will be added to the input filename to denote RHT output.  
xyt_suffix = '_xyt' 

# Limits ouput to 10^DIGITS files per input (_xyt00.fits to _xyt99.fits)
DIGITS = 2 

# User Interface
# Width of some displayed text objects
TEXTWIDTH = 70 

# Displays a progress bar, at a minor cost to speed.
PROGRESS = True 

# Displays information that would be helpful to developers and advanced users.
DEBUG = True 

# Allows processing of larger files than RAM alone would allow
# Give the program permission to create a temporary directory for RHT data.
BUFFER = True

# Single precision
DTYPE = np.float32

# Maximum number of bytes allowed for a single buffer file. There can be multiple buffer files.
FILECAP = int(5e8)


# Excluded Data Types
BAD_0 = False
BAD_INF = True
BAD_Neg = False 

# Timers #______________Should be put into a timer object
start_time = None
stop_time = None

#-----------------------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------------------
def announcement(strings):
    result = ''
    if type(strings) == list:
        strings.append('*'*TEXTWIDTH)
        strings.insert(0, '*'*TEXTWIDTH)
        result = '\n'.join(str.center(str(s), TEXTWIDTH, ' ') for s in strings) 
    elif type(strings) == str:
        result = announcement(strings.split('\n'))
    else:
        result = announcement(str(strings))
    return result

def announce(strings):
    print(announcement(strings))

def update_progress(progress, message='Progress:', final_message='Finished:'):
    # Create progress meter that looks like: 
    # message + ' ' + '[' + '#'*p + ' '*(length-p) + ']' + time_message
    if not PROGRESS:
        # Allows time-sensitive jobs to be completed without timing overhead
        return 

    if not 0.0 <= progress <= 1.0:
        # Fast fail for values outside the allowed range
        raise ValueError('Progress value outside allowed value in update_progress') 

    #TODO_________________Slow Global Implementation
    global start_time
    global stop_time 

    # First call
    if 0.0 == progress:
        start_time = time.time()
        stop_time = None
        return

    # Second call
    elif stop_time is None:
        stop_time = start_time + (time.time() - start_time)/progress

    # Randomly callable re-calibration
    elif np.random.rand() > 0.98: 
        stop_time = start_time + (time.time() - start_time)/progress

    # Normal Call with Progress
    sec_remaining = int(stop_time - time.time())
    if sec_remaining >= 60:
        time_message = ' < ' + str(sec_remaining//60  +1) + 'min'
    else:
        time_message = ' < ' + str(sec_remaining +1) + 'sec'

    length = int(0.55 * TEXTWIDTH)
    messlen = TEXTWIDTH-(length+3)-len(time_message)
    message = str.ljust(message, messlen)[:messlen]

    p = int(length*progress/1.0) 
    sys.stdout.write('\r{2} [{0}{1}]{3}'.format('#'*p, ' '*(length-p), message, time_message))
    sys.stdout.flush()

    # Final call
    if p == length:
        total = int(time.time()-start_time)
        if total > 60:
            time_message = ' ' + str(total//60) + 'min'
        else:
            time_message = ' ' + str(total) + 'sec'
        
        final_offset = TEXTWIDTH-len(time_message)
        final_message = str.ljust(final_message, final_offset)[:final_offset]
        sys.stdout.write('\r{0}{1}'.format(final_message, time_message))
        sys.stdout.flush()
        start_time = None
        stop_time = None
        print('')

#-----------------------------------------------------------------------------------------
# Naming Conventions and Converisons
#-----------------------------------------------------------------------------------------

def filename_from_path(filepath):
    # Maintains all characters in path except for those after and including the last period
    return os.path.basename('.'.join( filepath.split('.')[ 0:filepath.count('.') ] ) ) 

def xyt_name_factory(filepath, wlen, smr, frac, original):
    # Returns the filename that _xyt output should have.
    # Will have the general behavior: filename_xyt00.format

    # filepath ~ dirname/name.fits
    # filename ~ dirname/name
    # fnmatch_string ~ name + xyt_suffix + ?? + xyt_format

    # Remove RHT-specific endings
    filename = filename_from_path(filepath)
    
    if OUTPUT == '.':
        dirname = os.path.dirname(os.path.abspath(filepath))
    else:
        dirname = OUTPUT
    
    fnmatch_string = filename + xyt_suffix + '?'*DIGITS + xyt_format 
    xyt_files = fnmatch.filter(os.listdir(dirname), fnmatch_string) 
    xyt_array = [None]*(10**DIGITS) 

    # Try to find a parameter match among existing files
    left = str.find(fnmatch_string, '?')
    for x in xyt_files:
        abs_x = os.path.join(dirname, x)

        if getXYT(abs_x, match_only={'WLEN':wlen, 'SMR':smr, 'FRAC':frac, 'ORIGINAL':original} ): #TODO ______________________________________#print 'Found _xyt file matching your input parameters!'

            return os.path.normpath(abs_x)
        else:
            xyt_array[int( x[left:(left+DIGITS)] )] = x
    
    # Try to find the lowest-numbered name that is unoccupied
    for i, y in enumerate(xyt_array):
        if y is None:
            # Print 'Found _xyt available for these parameters!'
            int_string = str.zfill(str(i), DIGITS)[:DIGITS] 
            xyt_filename = filename+ xyt_suffix+ int_string+ xyt_format
            return os.path.normpath(os.path.join(dirname, xyt_filename))
    
    # Failure: No match and no available output slots
    xyt_filename = str.replace(fnmatch_string, '?', '0')
    print('In xyt_filename(): No existing ouput matches the input parameters and no namespace is available')
    print('Overwrite ' + xyt_filename + '?..') 
    choice = input(' [y]/n/'+'0'*(DIGITS-1)+'x')
    if len(choice) == 0 or choice == 'y':
        return os.path.normpath(os.path.join(dirname, xyt_filename))
    elif choice != 'n':
        int_string = str.zfill(str(int(choice)), DIGITS)[:DIGITS] 
        xyt_filename = filename+ xyt_suffix+ int_string+ xyt_format
        return os.path.normpath(os.path.join(dirname, xyt_filename)) 
    else:
        raise RuntimeError('In xyt_filename(): No existing ouput matches the input parameters and no namespace is available')

#-----------------------------------------------------------------------------------------
# Image Processing Functions
#-----------------------------------------------------------------------------------------
def is_valid_file(filepath):
    '''
    filepath: Potentially a string path to a source file for the RHT

    return: Boolean, True ONLY when the data might have rht() applied successfully
    '''

    excluded_file_endings = [] #TODO___More Endings

    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False
    
    excluded_file_content = ['_xyt', '_backproj', '_spectrum', '_plot', '_result'] 
    if any([e in filepath for e in excluded_file_content]):
        return False

    return True

def ntheta_w(w=WLEN):
    # Returns the number of theta bins in each Hthet array

    # Linearly proportional to wlen
    return int(math.ceil( np.pi*(w-1)/np.sqrt(2.0) ))  

# Saves the data into the given xyt_filename, depending upon filetype. Supports .fits and .npz currently
def putXYT(filepath, xyt_filename, hi, hj, hthets, wlen, smr, frac, original, backproj=None, compressed=True):

    if xyt_filename.endswith('.npz'):
        # IMPLEMENTATION1: Zipped Numpy arrays of Data #TODO _______________________________________ALWAYS BE CAREFUL WITH HEADER VARS
        if compressed:

            save = np.savez_compressed  
        else:
            save = np.savez
        if backproj is None:
            save(xyt_filename, hi=hi, hj=hj, hthets=hthets, wlen=wlen, smr=smr, frac=frac, original=original, ntheta=hthets.shape[1])
        else:
            save(xyt_filename, hi=hi, hj=hj, hthets=hthets, wlen=wlen, smr=smr, frac=frac, original=original, ntheta=hthets.shape[1], backproj=backproj)



    elif xyt_filename.endswith('.fits'):
        # IMPLEMENTATION2: FITS Table File
        Hi = fits.Column(name='hi', format='1I', array=hi)
        Hj = fits.Column(name='hj', format='1I', array=hj)
        ntheta = hthets.shape[1]
        Hthets = fits.Column(name='hthets', format=str(int(ntheta))+'E', array=hthets)
        cols = fits.ColDefs([Hi, Hj, Hthets])
        tbhdu = fits.BinTableHDU.from_columns(cols)

        # Header Values for RHT Parameters
        prihdr = fits.Header()
        prihdr['WLEN'] = wlen 
        prihdr['SMR'] = smr
        prihdr['FRAC'] = frac
        prihdr['ORIGINAL'] = original

        # Other Header Values
        prihdr['NTHETA'] = ntheta
        
        """
        Adding RA, DEC and other possible header values to your new header
        
        First, the old header is loaded in from filepath.
        
        You can then overwrite your desired header information by 
        adding/removing the keywords below. 
        """
        
        # Old header
        my_header = fits.getheader(filepath)
        
        # If you do not want header keywords from your old header, make this an empty list.
        # If you do, just input them as strings: ['CTYPE1', 'CRVAL1'] etc.
        header_keywords = []
        
        if len(header_keywords) > 0:
            for keyword in header_keywords:
                
                if keyword not in my_header:
                    print("Provided header keyword not in your old header. Please adjust variable header_keywords in function putXYT. Exiting...")
                    sys.exit()
                
                prihdr[keyword] = my_header[keyword]
            
        # Whole FITS File
        prihdu = fits.PrimaryHDU(data=backproj, header=prihdr)
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(xyt_filename, output_verify='silentfix', overwrite=True, checksum=True)

        #TODO__________________Compress Fits Files After Saving

    else:
        raise ValueError('Supported output filetypes in putXYT include: .npz and .fits only')

def getXYT(xyt_filename, match_only=False):    
    # Read in a .fits or .npz file containing the output of the RHT.
    # If match_only is given, and a dictionary of Keys:
    #     This will return whether ALL keys are found in the data of the given file 
    # Else:
    #     This will return the image coordinates of significant linearity, and the theta power spectrum at those coords. 
    #     This will return as two integer arrays of some_length, and an ntheta*some_length array of theta power

    if not os.path.isfile(xyt_filename):
        # Fast Failure Case - This file does not exist.
        if match_only:
            return False
        else:
            raise ValueError('Input xyt_filename in getXYT matches no existing file')
    else:
        # Attempts to extract header information for Matching, or else the data itself
        if xyt_filename.endswith('.npz'):
            # Allows very large files to be read in.
            data = np.load(xyt_filename, mmap_mode='r')
            if match_only:
                try:
                    return all([ match_only[x] == data[str.lower(x)] for x in list(match_only.keys()) ])
                except KeyError:
                    return False
            Hi = data['hi']
            Hj = data['hj']
            Hthets = data['hthets']

        elif xyt_filename.endswith('.fits'):
            hdu_list = fits.open(xyt_filename, mode='readonly', memmap=True, save_backup=False, checksum=True) #Allows for reading in very large files!
            header = hdu_list[0].header
            if match_only:
                try:
                    return all([ match_only[x] == header[str.upper(x)] for x in list(match_only.keys()) ])
                except KeyError:
                    return False
            data = hdu_list[1].data
            Hi = data['hi'] 
            Hj = data['hj'] 
            Hthets = data['hthets']

        else:
            raise ValueError('Supported input types in getXYT include .npz and .fits only')

    rebuild = None
    # Formats output properly
    if rebuild and filepath is not None:
        # Can recreate an entire 3D array of mostly 0s.
        data = getData(filepath)
        datay, datax = data.shape
        ntheta = Hthets[0].shape  
        if BUFFER:
            xyt = np.memmap(tempfile.TemporaryFile(), dtype=DTYPE, mode='w+', shape=(datay, datax, ntheta))
            xyt.fill(0.0)
        else:
            print('Warning: Reconstructing very large array in memory! Set BUFFER to True!')  
            xyt = np.zeros((datay, datax, ntheta))
        coords = list(zip(Hj, Hi))
        for c in range(len(coords)):
            j,i = coords[c]
            xyt[j,i,:] = Hthets[c]
        return xyt
    else:
        # Returns the sparse, memory mapped form only.
        return Hi, Hj, Hthets   


def bad_pixels(data):
    # Returns an array of the same shape as data
    # NaN values MUST ALWAYS be considered bad.
    # Bad values become 1, all else become 0
    data = np.array(data, np.float) #TODO________________________Double Check This?
    
    # IMPLEMENTATION1: Do Comparisons which are VERY different depending on boolean choices .
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
        # IMPLEMENTATION3: Give up?
        print('Unable to properly mask data in bad_pixels()...')
        return data.astype(np.bool)

def all_within_diameter_are_good(data, diameter):
    assert diameter%2
    r = int(np.int(diameter/2))

    # Base case, 'assume all pixels are bad'
    mask = np.zeros_like(data)

    # Edge case, 'any pixel not within r of the edge might be ok'
    datay, datax = data.shape
    mask[r:datay-r, r:datax-r] = 1

    # Identifiably bad case, 'all pixels within r of me are not bad'
    circle = circ_kern(diameter)
    y_arr, x_arr = np.nonzero(circle)
    y_arr = y_arr - r
    x_arr = x_arr - r

    # IMPLEMENTATION1: Zero any mask pixel within r of a bad pixel
    update_progress(0.0)
    coords = list(zip(*np.nonzero(bad_pixels(data))))
    N = len(coords)
    for c in range(N):    
        j,i = coords[c]
        x = (x_arr + i).astype(np.int).clip(0, datax-1)
        y = (y_arr + j).astype(np.int).clip(0, datay-1)
        mask[y, x] = 0 
        update_progress((c+1)/float(N), message='Masking:', final_message='Finished Masking:') 
    '''
    #IMPLEMENTATION2: For each good pixel, 'Not Any Bad pixels near me'
    update_progress(0.0)
    coords = zip(*np.nonzero(mask))
    for c in range(len(coords)):
        j,i = coords[c]
        x = (x_arr + i).astype(np.int).clip(0, datax-1)
        y = (y_arr + j).astype(np.int).clip(0, datay-1)
        mask[j][i] = np.logical_not(np.any(bad_pixels( data[y, x] )))
        update_progress((c+1)/float(N), message='Masking:', final_message='Finished Masking:') 
    '''
    return mask 

def getData(filepath):
    # Reads in data for images from various sources
    # Supports .fits, .npy, and PIL formats
    
    try:
        # Reading Data
        if filepath.endswith('.fits'):
            # Fits file handling
            hdu = fits.open(filepath, memmap=True)[0] #TODO___________________Assumes all data is in first HDU
            data = hdu.data 

        elif filepath.endswith('.npy'):
            # Numpy file handling
            data = np.load(filepath, mmap_mode='r') 
        
        elif filepath.endswith('.npz'):
            data = np.load(filepath, mmap_mode='r')[0] #TODO___________________Assumes data in first ndarray is 2D

        else:
            data = scipy.ndimage.imread(filepath, flatten=True)[::-1] #Makes B/W array, reversing y-coords 

    except:
        # Failure Reading Data
        print('Failure in getData({})... Returning'.format(filepath))
        return None

    return data

def getMask(data, smr=SMR, wlen=WLEN):
    # Makes proper masks for images from data
    # smr_mask masks any pixel within smr of any bad pixels, and the edge
    # wlen_mask masks any pixel within wlen of any bad pixels, and the edge
    
    # Cuts away smr radius from bads, then wlen from bads 
    smr_mask = all_within_diameter_are_good(data, 2*smr+1)
    nans = np.empty(data.shape, dtype=np.float).fill(np.nan)
    wlen_mask = all_within_diameter_are_good( np.where(smr_mask, data, 
        nans), wlen)
    return smr_mask, wlen_mask

# Performs a circle-cut of given diameter on inkernel.
# Outkernel is 0 anywhere outside the window.   
def circ_kern(diameter):
    assert diameter%2
    r = diameter//2 #int(np.floor(diameter/2))
    mnvals = np.indices((diameter, diameter)) - r
    rads = np.hypot(mnvals[0], mnvals[1])
    return np.less_equal(rads, r).astype(np.int)

# Unsharp mask. Returns binary data.
def umask(data, radius, smr_mask=None):
    assert data.ndim == 2

    kernel = circ_kern(2*radius+1)
    outdata = scipy.ndimage.filters.correlate(data, kernel) 
    
    # Correlation is the same as convolution here because kernel is symmetric 
    # Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(kernel)
    subtr_data = data - outdata/kernweight
    
    # Convert to binary data
    bindata = np.greater(subtr_data, 0.0)
    if smr_mask is None:
        return bindata
    else:
        return np.logical_and(smr_mask, bindata)


def fast_hough(in_arr, xyt):  

    assert in_arr.ndim == 2 
    assert xyt.ndim == 3
    assert in_arr.shape[0] == xyt.shape[0]
    assert in_arr.shape[1] == xyt.shape[1]

    # IMPLEMENTATION0: Let python figure out the implementation. (FASTEST)
    return np.einsum('ijk,ij', xyt, in_arr)
    
    '''
    if hout == None:
        return np.einsum('ijk,ij', xyt, in_arr) #, dtype=np.int) 
    else:
        assert hout.ndim == 1
        assert hout.shape[0] == xyt.shape[2]
        np.einsum('ijk,ij', xyt, in_arr, out=hout)
    '''

    # IMPLEMENTATION1: Copy 2D array into 3D stack, and multiply by other stack (SLOW)
    # cube = np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)*xyt
     
    # IMPLEMENTATION2: Broadcast 2D array against 3D stack and multiply (FAST)
    # cube = np.multiply( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt).astype(np.float, copy=False)

    # IMPLEMENTATION3: Broadcast 2D array against 3D stack and AND them together (VERY FAST)
    # assert in_arr.dtype == np.bool_ 
    # assert xyt.dtype == np.bool_  
    # cube = np.logical_and( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt)
        
    # return np.sum(np.sum( cube , axis=0, dtype=np.int), axis=0, dtype=np.float) #WORKS FAST AND DIVIDES PROPERLY
    # return np.sum(cube, axis=(1,0), dtype=np.int)

def houghnew(image, cos_theta, sin_theta):
    assert image.ndim == 2 
    assert cos_theta.ndim == 1
    assert sin_theta.ndim == 1
    assert len(cos_theta) == len(sin_theta)
    assert image.shape[0] == image.shape[1]

    # Midpoint is wlen/2
    wmid = image.shape[0]//2 

    # Compute the distance from each cell.
    nr_bins = np.ceil(np.hypot(*image.shape))

    # Allocate the output data.
    out = np.zeros((int(nr_bins), len(cos_theta)), dtype=np.int)

    # Find the indices of the non-zero values in the input data.
    y, x = np.nonzero(image)

    # x and y can be large, so we can't just broadcast to 2D arrays as we may run out of memory. 
    # Instead we process one vertical slice at a time.
    for i, (cT, sT) in enumerate(zip(cos_theta, sin_theta)):

        # Compute the base distances
        distances = (x - wmid) * cT + (y - wmid) * sT

        # Round the distances to the nearest integer and shift them to a nonzero bin.
        shifted = np.round(distances) + nr_bins/2

        # Cast the shifted values to ints to use as indices
        indices = shifted.astype(np.int)
        
        # Use bin count to accumulate the HT coefficients
        bincount = np.bincount(indices) 

        # Assign the proper values to the out array
        out[:len(bincount), i] = bincount

    return out[np.int(nr_bins/2), :]


def all_thetas(wlen, theta, original):
    assert theta.ndim == 1 
    assert wlen%2

    # Initialize a circular window of ones
    window = circ_kern(wlen)
    assert window.shape[0] == window.shape[1]
    if not original:
        window[:,:wlen//2] = 0
    
    # Precompute the sin and cos of the angles.
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Makes prism; output has dimensions (y, x, theta).
    ntheta = len(theta)
    
    #outshape = (wlen, wlen, ntheta)
    out = np.zeros(window.shape+(ntheta,), np.int)
    coords = list(zip( *np.nonzero(window)))
    for (j, i) in coords:
        # At each x/y value, create new single-pixel data.
        w_1 = np.zeros_like(window)
        w_1[j,i] = 1
        out[j, i, :] = houghnew(w_1, cos_theta, sin_theta)

    if not original:
        out[:,:,ntheta//2:] = out[::-1,::-1,ntheta//2:] 
        out[:wlen//2+1,:,ntheta//2] = 0
        out[wlen//2:,:,0] = 0

    return out 


def theta_rht(theta_array, original, uv=False):
    # Maps an XYT cube into a 2D Array of angles- weighted by their significance.
    if original:
        thetas = np.linspace(0.0, np.pi, len(theta_array), endpoint=False, retstep=False)
        ys = theta_array*np.sin(2.0*thetas)
        xs = theta_array*np.cos(2.0*thetas)    
        
        # Clark, Peek, & Putman: Equation (7)
        rough_angle = 0.5*np.arctan2(np.sum(ys), np.sum(xs))
        
        # Clark, Peek, & Putman: Equation (8)
        angle = np.pi-math.fmod(rough_angle+np.pi, np.pi)
        
    else:
        thetas = np.linspace(0.0, 2*np.pi, len(theta_array), endpoint=False, retstep=False)
        ys = theta_array*np.sin(thetas)
        xs = theta_array*np.cos(thetas)    
        angle = np.arctan2(np.sum(ys), np.sum(xs)) 
        
    if not uv:
        # Returns the <theta>_rht as described by Clark, Peek, & Putman for a given array.
        return angle
    else:
        # Maps an array of theta power to an angle, and the components of one unit vector.
        # Designed for use with plt.quiver()
        return angle, np.cos(angle), np.sin(angle)


def buffershape(ntheta, filesize=FILECAP):
    # Shape of maximum sized array that can fit into a single buffer file. 
    ntheta = int(ntheta)
    filesize = int(filesize)
    if not 0 < filesize <= FILECAP:
        print('Chosen buffer size exceeds existing limit. Reset to', str(FILECAP), 'Bytes')
        filesize = FILECAP

    bits_per_element_in_bits = np.dtype(DTYPE).itemsize
    bits_per_file_in_bits = filesize*8
    elements_per_file_in_elements = int(bits_per_file_in_bits // bits_per_element_in_bits)
    length_in_elements = int(elements_per_file_in_elements // ntheta)
    if length_in_elements <= 0:
        print('In buffershape, ntheta has forced your buffer size to become larger than', filesize, 'Bytes')
        length_in_elements = 1

    return (length_in_elements, ntheta) 

def concat_along_axis_0(memmap_list):
    # Combines memmap objects of the same shape, except along axis 0, 
    # by leaving them all on disk and appending them sequentially.
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
        # IMPLEMENTATION2: Append data to first given memmaped file, then delete and repeat
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
            
            # Depends on offset correctly allocating new space at end of file
            c[:,...] = b[:,...]
            b.flush()         
            b.close()        
            del b
            return c 

        return reduce(append_memmaps, others, initializer=seed)

def window_step(data, wlen, frac, smr, original, smr_mask, wlen_mask, 
        xyt_filename, message, filepath): 

    assert frac == float(frac) 
    assert 0 <= frac <= 1
    
    assert wlen == int(wlen)
    assert wlen > 0 
    
    # wlen must be an odd number to ensure the circle has a single-pixel center.
    assert wlen%2

    assert smr == int(smr)
    assert smr > 0
    
    # Needed values
    r = wlen//2 
    ntheta = ntheta_w(wlen)
    
    if original:
        theta, dtheta = np.linspace(0.0, np.pi, ntheta, endpoint=False, 
                retstep=True)        
    else:
        # For the dRHT, we maintain ntheta by doubling dtheta.
        theta, dtheta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False, 
                retstep=True)

    # Cylinder of all lit pixels along a theta value
    xyt = all_thetas(wlen=wlen, theta=theta, original=original) 
    xyt.setflags(write=0)
    
    # Unsharp masks the whole data set
    masked_udata = umask(data=data, radius=smr, smr_mask=smr_mask)
    masked_udata.setflags(write=0)

    # Hough transform of same-sized circular window of 1's
    h1 = fast_hough(circ_kern(wlen), xyt)
    h1.setflags(write=0)

    # Local function calls are faster than globals
    Hthets = []
    Hi = []
    Hj = []
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    nptruediv = np.true_divide
    npge = np.greater_equal
    
    # Bonus Backprojection Creation
    backproj = np.zeros_like(data)

    if BUFFER:
        # Preparing to write hout to file during operation so it does not over-fill RAM.
        temp_dir = tempfile.mkdtemp()
        
        # List of memmap objects.
        temp_files = []
        buffer_shape = buffershape(ntheta)
        def next_temp_filename():
            return os.path.join(temp_dir, 'rht'+ str(len(temp_files)) + '.dat')
        #print 'Temporary files in:', temp_dir 

    # Number of RHT operations that will be performed, and their coordinates
    update_progress(0.0)
    coords = list(zip( *np.nonzero( wlen_mask)))
    N = len(coords)
    for c in range(N):
        j,i = coords[c]
        h = fast_hough(masked_udata[j-r:j+r+1, i-r:i+r+1], xyt)

        # Original RHT Implementation Subtracts Threshold From All Theta-Power Spectrums
        hout = nptruediv(h, h1) - frac
        hout *= npge(hout, 0.0)
        # Deprecated Implementation Leaves Theta-Power Spectrum AS IS
        #hout = nptruediv(h, h1)
        #hout *= npge(hout, frac)

        if np.any(hout):
            htapp(hout)
            hiapp(i)
            hjapp(j)
            backproj[j][i] = np.sum(hout) 

            if BUFFER and len(Hthets) == buffer_shape[0]:
                # Creates full memmap object
                temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=buffer_shape ))
                
                # Convert list to array
                theta_array = np.array(Hthets, dtype=DTYPE)
                
                # Write array to last memmapped object in list
                temp_files[-1][:] = theta_array[:]
                
                # Reset Hthets
                Hthets = []

        update_progress((c+1)/float(N), message=message, final_message=message)
        #End

    if not BUFFER:
        # Save data
        putXYT(filepath, xyt_filename, np.array(Hi), np.array(Hj), np.array(Hthets), wlen, smr, frac, original=original, backproj=np.divide(backproj, np.amax(backproj)) ) 
        return True 

    else:
        if len(Hthets) > 0:
            # Create small memmap object
            temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=(len(Hthets), ntheta)  ))
            
            # Write array to last memmapped object in list
            theta_array = np.array(Hthets, dtype=DTYPE)
            temp_files[-1][:] = theta_array[:]

        #print 'Converting list of buffered hthet arrays into final XYT cube'
        # Combine memmap objects sequentially
        converted_hthets = concat_along_axis_0(temp_files)
        converted_hthets.flush()
        putXYT(filepath, xyt_filename, np.array(Hi), np.array(Hj), converted_hthets, wlen, smr, frac, original=original, backproj=np.divide(backproj, np.amax(backproj)) ) #Saves data
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
                print('Failed to delete temporary files:', path) 

        shutil.rmtree(temp_dir, ignore_errors=False, onerror=rmtree_failue)
        return True

#-----------------------------------------------------------------------------------------
# Interactive Functions
#-----------------------------------------------------------------------------------------

def rht(filepath, force=False, original=ORIGINAL, wlen=WLEN, frac=FRAC, 
        smr=SMR, data=None):
    """
    filepath: String path to source data, which will have the Rolling Hough
        Transform applied - if data is input (see below) then filepath is not
        read but is just used to construct the name of the output filename

    force: Boolean indicating if rht() should still be run, even when output
        exists for these inputs

    original: Boolean if one should use the original Rolling Hough Transform

    wlen: Diameter of a 'window' to be evaluated at one time

    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit
        up' to be counted

    smr: Integer radius of gaussian smoothing kernel to be applied to an data

    data: Input data array (image) - alternative to giving filepath as source
        of the data
    
    Saves:
        X-Y-Theta Power Arrays
        Backprojection

    return: Boolean, if the function succeeded
    """

    assert frac == float(frac) 
    assert 0 <= frac <= 1 
    
    assert wlen == int(wlen) 
    assert wlen > 0 
    assert wlen%2

    assert smr == int(smr) 
    assert smr > 0 

    if not is_valid_file(filepath):
        # Check to see if a file should have the rht applied to it.
        print('Invalid filepath encountered in rht('+filepath+')...')
        return False

    try:

        xyt_filename = xyt_name_factory(filepath, wlen=wlen, smr=smr, 
                frac=frac, original=original)
        
        if (not force) and os.path.isfile(xyt_filename):
            # If the program recognizes that the RHT has already been
            # completed, it will not rerun.  This can overridden by setting
            # the 'force' flag.
            return True


        if data is None:
            print('1/4:: Retrieving Data from:', filepath)
            data = getData(filepath)
        else:
            print('1/4:: Getting Mask for Data')
        smr_mask, wlen_mask = getMask(data, smr=smr, wlen=wlen)
        datay, datax = data.shape

        print('2/4:: Size: {} x {}, Wlen: {}, Smr: {}, Frac: {},'.format(
                datax,datay,wlen,smr,frac), 
                'Standard (half-polar) RHT:'.format(original))        
        message = '3/4:: Running RHT...'

        success = window_step(data=data, wlen=wlen, frac=frac, smr=smr, 
                original=original, smr_mask=smr_mask, wlen_mask=wlen_mask, 
                xyt_filename=xyt_filename, message=message, 
                filepath = filepath)

        print('4/4:: Successfully Saved Data As', xyt_filename)
        return success
    
    except:
        raise #__________________________________________________________________________________________________________ Raise
        return False

def interpret(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR, original=ORIGINAL):

    '''
    filepath: String path to source data, which will have the Rolling Hough Transform applied

    force: Boolean indicating if rht() should still be run, even when output exists for these inputs
    original: Boolean if one should use the original Rolling Hough Transform

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an data
    
    Displays:
        ?

    Saves:
        ?

    return: Boolean, if the function succeeded
    '''


    print('viewer() is currently in disrepair! Exiting to avoid unpleasant results!')
    return False 

    # Make sure relevant files are present.
    filename = filename_from_path(filepath)
    xyt_filename = filename + '_xyt.fits'  
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'
    required_files = [backproj_filename, spectrum_filename, xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing or force:
        # Interprets file, after clearing old output, since that needs to be done.
        for f in required_files:
            try:
                # Try deleting obsolete output.
                os.remove(f)
            except:
                # Assume it's not there.
                continue 
        interpret(filepath, force=force, wlen=wlen, frac=frac, smr=smr, original=original) 

    # Load in relevant files and data
    data = getData(filepath)
    backproj = np.load(backproj_filename).astype(np.float)
    spectrum = np.load(spectrum_filename).astype(np.float)
    hi, hj, hthets = getXYT(xyt_filename)

    # Gather parameters from that data
    ntheta = len(spectrum)
    datay, datax = data.shape

    # Produce Specific Plots
    masked_udata = umask(data, smr)
    log = np.log(np.where( np.isfinite(data), data, np.ones_like( data) ))
    U = np.zeros_like(hi)
    V = np.zeros_like(hj)
    C = np.zeros((len(U)), dtype=np.float)
    coords = list(zip(hi, hj))
    for c in range(len(coords)):
        C[c], U[c], V[c] = theta_rht(hthets[c], original, uv=True)
    C *= np.isfinite(C)
    if original:
        C /= np.pi
    else:
        C /= 2*np.pi 
    np.clip(C, 0.0, 1.0, out=C)

    # Define Convenience Functions
    def cleanup():
        plt.clf()
        plt.cla()
        plt.close()

    #---------------------------------------------------------------------- PLOT 1
    print('Plotting Whole Figure...')
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
    print('Backprojecting...')

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
    print('Linearity')
    
    if original:
        modified_spectrum = np.true_divide(np.append(spectrum, spectrum), 2.0) 
        plt.polar(np.linspace(0.0, 2*np.pi, num=ntheta, endpoint=False), modified_spectrum)
    else:
        plt.polar(np.linspace(0.0, 2*np.pi, num=ntheta, endpoint=False), spectrum)
    plt.show()
    cleanup()
    

    #---------------------------------------------------------------------- PLOTS END
    #Clear all and exit successfully
    cleanup()
    return True



def main(source=None, display=False, force=False, drht=False, wlen=WLEN, 
        frac=FRAC, smr=SMR, data=None):
    """
    source: A filename, or the name of a directory containing files to transform

    display: Boolean flag determining if the input is to be interpreted and
        displayed
    
    force: Boolean flag determining if rht() will be run, when output already
        exists

    wlen: Diameter of a 'window' to be evaluated at one time
    
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit
        up' to be counted
    
    smr: Integer radius of gaussian smoothing kernel to be applied to an data

    return: Boolean, if the function succeeded
    """

    # Setting 'drht' to True means that the internal parameter 'original' is False.
    if drht == True:
        original = False
    else:
        original = True
    
    # Ensure that the input is a non-None, non-Empty string
    while source is None or type(source) != str or len(source)==0:
        try:
            source = input('Source:')
        except:
            source = None
    
    # Interpret whether the Input is a file or directory, excluding all else
    pathlist = []
    if os.path.isfile(source):
        # Input is a file.
        pathlist.append(source)
    elif os.path.isdir(source):
        # Input is a directory. 
        for obj in os.listdir(source):
            obj_path = os.path.join(source, obj)
            if os.path.isfile(obj_path):
                pathlist.append(obj_path)
    else:
        # Input is neither a file nor a directory.
        print('Invalid source encountered in main(); must be file or directory.')
        return False

    pathlist = list(filter(is_valid_file, pathlist))
    if len(list(pathlist)) == 0:
        print('Invalid source encountered in main(); no valid images found.')
        return False

    # Run RHT over all valid inputs. 
    announce(['Fast Rolling Hough Transform by Susan Clark', 'Started for: '+source])

    summary = [] #TODO__________________________________________________________________________________ batch progress bar

    for path in pathlist:
        success = True
        try:
            if (display):
                success = viewer(path, force=force, original=original, 
                        wlen=wlen, frac=frac, smr=smr)
            else:
                success = rht(path, force=force, original=original, wlen=wlen,
                        frac=frac, smr=smr, data=data)
        except:
            success = False
            if DEBUG:
                raise 
        finally:
            if success:
                summary.append(path+': Passed')
            else:
                summary.append(path+': Failed')
    summary.append('Complete!')
    announce(summary)
    return True

#------------------------------------------------------------------------------
# Initialization 3 of 3: Precomputed Objects
#------------------------------------------------------------------------------


        
#------------------------------------------------------------------------------
# Command Line Mode
#------------------------------------------------------------------------------
def cli():
    parser = ArgumentParser(description="Run Rolling Hough Transform on 1+ FITS files",
        usage='%(prog)s [options] file(s)',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('files',nargs='+',help="FITS file(s)")
    parser.add_argument('-f','--force', action="store_true",
        help="force overwriting of existing output _xyt.fits files")
    parser.add_argument('-w','--wlen',default=WLEN,type=int,
        help="Window diameter in pixels")
    parser.add_argument('-s','--smr',default=SMR,type=int,
        help="Smoothing radius for unsharp mask in pixels")
    parser.add_argument('-t','--thresh',default=FRAC,type=float,
        help="Fraction (Threshold) of a given theta that must be 'lit up' to be counted")
    parser.add_argument('-d','--drht',action="store_true",
        help="Compute Directional RHT (full polar)")
    parser.add_argument('--version',action='version',version='%(prog)s 1.0')

    if len(sys.argv) == 1: # no arguments given, so add -h to get help msg
        sys.argv.append('-h')
    args = parser.parse_args()

    for f in args.files: # loop over input files
        main(source=f, force=args.force, wlen=args.wlen,
            frac=args.thresh, smr=args.smr, drht=args.drht)
    sys.exit()

if __name__ == "__main__":
    cli()

#------------------------------------------------------------------------------
#Attribution
#------------------------------------------------------------------------------

# This is the Rolling Hough Transform, described in Clark, Peek, & Putman
# 2014, ApJ 789, 82 (arXiv:1312.1338).  If use of the RHT leads to a
# publication, please cite the above.  Modifications to the RHT have been made
# by Lowell Schudel, les2185@columbia.edu, CC'16.
