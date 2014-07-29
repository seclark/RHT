#!/usr/bin/python
#FAST ROLLING HOUGH TRANSFORM
#Susan Clark, Lowell Schudel

#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------
from __future__ import division #Must be first line of code in the file
from astropy.io import fits
#from scipy.stats import norm
#from mpl_toolkits.mplot3d import Axes3D
#from mayavi import mlab
#from astropy import wcs
import numpy as np
import scipy.ndimage
import math
import os
import matplotlib.pyplot as plt
import sys
import string
import tempfile 
import shutil
import time 

#-----------------------------------------------------------------------------------------
#Initialization 1 of 3: Calculation Parameters
#-----------------------------------------------------------------------------------------

WLEN = 55 #101.0 #Diameter of a 'window' to be evaluated at one time
FRAC = 0.70 #0.70 #fraction (percent) of one angle that must be 'lit up' to be counted
SMR = 15.0 #smoothing radius of unsharp mask function

#-----------------------------------------------------------------------------------------
#Initialization 2 of 3: Runtime Parameters
#-----------------------------------------------------------------------------------------

TEXTWIDTH = 70 #Width of some displayed text objects

#Allows processing of larger files than RAM alone would allow
BUFFER = True #Gives the program permission to create a temporary directory for RHT data
DTYPE = np.float32 #Single precision
FILECAP = int(5e8) #Maximum number of BYTES allowed for a SINGLE buffer file. THERE CAN BE MULTIPLE BUFFER FILES!

#Excluded Data Types _______#TODO
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

def update_progress(progress, message='Progress:'):
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

    elif np.random.rand() > 0.95: #Randomly callable re-calibration
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
        start_time = None
        stop_time = None
        print ''

#-----------------------------------------------------------------------------------------
#Image Processing Functions
#-----------------------------------------------------------------------------------------
def is_valid_file(filepath):
    '''
    filepath: Potentially a string path to a source data

    return: Boolean, True ONLY when the data could have rht() applied successfully
    '''
    excluded_file_endings = ['_xyt.npz', '_backproj.npy', '_spectrum.npy', '_plot.png', '_result.png'] #TODO___More Endings
    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False
    
    excluded_file_content = ['_xyt', '_backproj', '_spectrum', '_plot', '_result'] #TODO___More Exclusions
    if any([e in filepath for e in excluded_file_content]):
        return False

    return True

def ntheta_w(w=WLEN):
    #Returns the number of theta bins in each Hthet array

    #Linearly proportional to wlen
    return int(math.ceil( np.pi*(w-1)/np.sqrt(2.0) ))  #TODO_________________________________ntheta

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
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        center_filename = filename+'_center.npy'
        np.save(center_filename, cutout)
        return center_filename
    else:
        return center(filepath, shape=(x//2,y//2))

def putXYT(xyt_filename, hi, hj, hthets, compressed=True): #TODO _______________________________________PARAMETERS
    if xyt_filename.endswith('.npz'):
        #IMPLEMENTATION1: Zipped Numpy arrays of Data
        if compressed:
            np.savez_compressed(xyt_filename, hi=hi, hj=hj, hthets=hthets)  #TODO _______________________________________HEADER VARS
        else:
            np.savez(xyt_filename, hi=hi, hj=hj, hthets=hthets)

    elif xyt_filename.endswith('.fits'):
        #IMPLEMENTATION2: FITS Table File
        Hi = fits.Column(name='hi', format='1I', array=hi)
        Hj = fits.Column(name='hj', format='1I', array=hj)

        Hthets = [fits.Column(name='hthets', format='E', array=hthets.T[i]) for i in xrange(len(hthets[0]))]
        
        Hthets.append(0, Hj)
        Hthets.append(0, Hi)
        cols = fits.ColDefs(Hthets)
        tbhdu = fits.BinTableHDU.from_columns(cols)

        #Header
        prihdr = fits.Header()
        prihdr['OBSERVER'] = 'Edwin Hubble' #TODO _______________________________________HEADER VARS
        
        #Whole FITS File
        prihdu = fits.PrimaryHDU(header=prihdr)
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(xyt_filename, output_verify='silentfix', clobber=True, checksum=True)


    else:
        raise ValueError('Supported output types in putXYT include .npz and .fits only')

def getXYT(xyt_filename, rebuild=False, filepath = None):    
    #Reads in a .npz file containing coordinate pairs in data space (hi, hj)
    #And Hough space arrays covering theta space at each of those points
    data = np.load(xyt_filename, mmap_mode='r') #Allows for reading in very large files!
    if rebuild and filepath is not None:
        #Can recreate an entire 3D array of mostly 0s
        data = getData(filepath)
        datay, datax = data.shape
        ntheta = data['hthets'][0].shape  

        if BUFFER:
            xyt = np.memmap(tempfile.TemporaryFile(), dtype=DTYPE, mode='w+', shape=(datay, datax, ntheta))
            xyt.fill(0.0)
        else:
            print 'Warning: Reconstructing very large array in memory...'
            print 'Size:', datay, 'x', datax, 'x', ntheta
            xyt = np.zeros((datay, datax, ntheta))

        coords = zip(data['hj'], data['hi'])
        for c in range(len(coords)):
            j,i = coords[c]
            xyt[j,i,:] = data['hthets'][c]
        return xyt
    else:
        #Returns the sparse, memory mapped form only
        return data['hi'], data['hj'], data['hthets']

 
def bad_pixels(data):
    #Returns an array of the same shape as data
    #NaN values MUST ALWAYS be considered bad.
    #Bad values become 1, all else become 0
    data = np.array(data, np.float)
    
    #IMPLEMENTATION1: Do Comparisons which are VERY different depending on boolean choices 
    if BAD_INF:
        if BAD_0:
            if BAD_Neg:
                return np.logical_or(np.logical_not(np.isfinite(data)), np.less_equal(0.0))
            else:    
                return np.logical_or( np.logical_not( np.isfinite(data) ), np.logical_not(data) )
        else:
            if BAD_Neg:
                return np.logical_or(np.logical_not(np.isfinite(data)), np.less(0.0))
            else:    
                return np.logical_not(np.isfinite(data))
    else:
        if BAD_0:
            if BAD_Neg:
                return np.logical_or(np.isnan(data), np.less_equal(0.0))
            else:    
                return np.logical_not(np.nan_to_num(data)) #(Nans or 0) ---> (0) ---> (1)
        else:
            if BAD_Neg:
                return np.logical_or(np.isnan(data), np.less(0.0))
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
    #IMPLEMENTATION3: Give up?
    print 'Unable to properly mask data in bad_pixels()...'
    return data.astype(np.bool)

def all_within_diameter_are_good(data, diameter):
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
    coords = zip(*np.nonzero(bad_pixels(data)))
    for c in range(len(coords)):    
        j,i = coords[c]
        x = (x_arr + i).astype(np.int).clip(0, datax-1)
        y = (y_arr + j).astype(np.int).clip(0, datay-1)
        mask[y, x] = 0
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
            data = np.load(filepath, mmap_mode='r') #Reads numpy files #TODO
        
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
        smr_mask = all_within_diameter_are_good(data, 2*smr)
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
    r = int(np.floor(diameter/2))
    mnvals = np.indices((diameter, diameter)) - r
    rads = np.hypot(mnvals[0], mnvals[1])
    return np.less_equal(rads, r)

#Unsharp mask. Returns binary data.
def umask(data, radius, smr_mask=None):
    kernel = circ_kern(2*radius)
    outdata = scipy.ndimage.filters.correlate(data, weights=kernel)

    #Correlation is the same as convolution here because kernel is symmetric 
    #Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(np.sum(kernel, axis=0), axis=0)
    subtr_data = data - outdata/kernweight
    
    #Convert to binary data
    bindata = np.greater(subtr_data, 0.0)
    if smr_mask == None:
        return bindata
    else:
        return np.where(smr_mask, bindata, smr_mask)

def fast_hough(in_arr, xyt, ntheta): #TODO_________________________________________#THIS IS ONE BOTTLENECK IN THE CODE

    assert in_arr.ndim == 2 
    assert xyt.ndim == 3
    assert in_arr.shape[0] == xyt.shape[0]
    assert in_arr.shape[1] == xyt.shape[1]

    #IMPLEMENTATION1: Copy 2D array into 3D stack, and multiply by other stack (SLOW)
    #return np.sum(np.sum(np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)*xyt, axis=0), axis=0)

    #IMPLEMENTATION2: Broadcast 2D array against 3D stack and multiply (FAST)
    return np.sum(np.sum( np.multiply( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt) , axis=0), axis=0)

    #IMPLEMENTATION3: Broadcast 2D array against 3D stack and AND them together (VERY FAST)
    #return np.sum(np.sum( np.logical_and( in_arr.reshape((in_arr.shape[0],in_arr.shape[1],1)), xyt) , axis=0), axis=0)

def houghnew(image, cos_theta, sin_theta):
    assert image.ndim == 2 
    assert cos_theta.ndim == 1
    assert sin_theta.ndim == 1
    assert len(cos_theta) == len(sin_theta)

    wy, wx = image.shape 
    wmid = np.floor(wx/2.0) #_____________________________________________TODO??

    # compute the vertical bins (the distances)
    nr_bins = np.ceil(np.hypot(*image.shape))

    # allocate the output data
    out = np.zeros((int(nr_bins), len(cos_theta)), dtype=np.bool_)

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
        bincount = np.bincount(indices) #TODO______________________ bincount method?

        # finally assign the proper values to the out array
        out[:len(bincount), i] = bincount
        #out.T[i] = bincount

    return out[np.floor(nr_bins/2), :]


def all_thetas(wlen, thetbins):
    window = circ_kern(wlen)
    window[:,:wlen//2] = 0
    wy, wx = window.shape #Parse x/y dimensions
    ntheta = len(thetbins) #Parse height in theta


    # precompute the sin and cos of the angles
    cos_theta = np.cos(thetbins)
    sin_theta = np.sin(thetbins)
    
    #Makes prism; output has dimensions (x, y, theta)
    out = np.zeros((wy, wx, ntheta), np.int)
    coords = zip( *np.nonzero(window))

    for (j, i) in coords:
        #At each x/y value, create new single-pixel data
        w_1 = np.zeros((wy, wx), np.float_)
        w_1[j,i] = 1
        out[j, i, :] = houghnew(w_1, cos_theta, sin_theta)

    out[:,:,ntheta//2:] = out[::-1,::-1,ntheta//2:] #TODO ____________________________________________________________________________________________ CYLINDER
    out[:wlen//2+1,:,ntheta//2] = 0
    out[wlen//2:,:,0] = 0

    return out 


def theta_rht(theta_array, uv=False):
    #Maps an XYT cube into a 2D Array of angles- weighted by their significance
    thetas = np.linspace(0.0, np.pi, len(theta_array), endpoint=False, retstep=False)
    ys = theta_array*np.sin(2.0*thetas)
    xs = theta_array*np.cos(2.0*thetas)    
    rough_angle = 0.5*np.arctan2(np.sum(ys), np.sum(xs)) #EQUATION (7)
    angle = np.pi-math.fmod(rough_angle+np.pi, np.pi) #EQUATION (8) #TODO __________________________________________Is this correct?
    if not uv:
        return angle
    else:
        return angle, np.cos(angle), np.sin(angle)
    #OR It can map all arrays to the vector (x, y) of theta power at one point
    #MADE FOR USE WITH plt.quiver() #TODO

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
        raise ValueError('Failed to buffer any data!') #TODO_______________________________Unhandled Exception

    elif len(memmap_list) == 1:
        return memmap_list[0]
    
    else:
        '''
        #IMPLEMENTATION1: Make a new large memmapped file and sequentially dump data in
        lengths = [memmap.shape[0] for memmap in memmap_list]
        shapes = [memmap.shape[1:] for memmap in memmap_list]
        assert all([x==shapes[0] for x in shapes[1:]])

        big_memmap = np.memmap(os.path.join(temp_dir, +'rht.dat'), dtype=DTYPE, mode='r+', shape=(sum(lengths), *shapes[0])  )   #TODO ______________________________________________________
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
            c = np.memmap(a.filename, dtype=DTYPE, mode='r+', offset=bytes_per_shape_in_bytes*a.shape[0], shape=b.shape )
            c[:,...] = b[:,...] #Depends on offset correctly allocating new space at end of file
            b.flush()         
            b.close()        
            del b
            return c 

        return reduce(append_memmaps, others, initializer=seed)

def window_step(data, wlen, frac, smr, theta, ntheta, smr_mask, wlen_mask, xyt_filename, message): 
    
    wcntr = int(np.floor(wlen/2))
    
    xyt = all_thetas(wlen, theta) #Cylinder of all theta values per point

    #Unsharp masks the whole data set
    masked_udata = umask(data, smr, smr_mask=smr_mask) 

    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(circ_kern(wlen), xyt, ntheta)

    #Local function calls are faster than globals
    Hthets = []
    Hi = []
    Hj = []
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    npsum = np.sum

    if not BUFFER:
        #Number of RHT operations that will be performed, and their coordinates
        coords = zip( *np.nonzero( wlen_mask))
        update_progress(0.0, message)
        N = len(coords)
        for c in range(N):
            j,i = coords[c]
            h = fast_hough(masked_udata[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1], xyt, ntheta)
            hout = np.divide(h, h1) #TODO____________________np.true_divide(h, h1)
            hout *= np.greater_equal(hout, frac)
            if np.any(hout):
                htapp(hout)
                hiapp(i)
                hjapp(j)
            update_progress((c+1)/float(N), message)
            #End
        putXYT(xyt_filename, np.array(Hi), np.array(Hj), np.array(Hthets)) #Saves data
        return True 

    else:
        #Preparing to write hout to file during operation so it does not over-fill RAM
        temp_dir = tempfile.mkdtemp()
        temp_files = [] #List of memmap objects
        buffer_shape = buffershape(ntheta) #variable_length, ntheta
        def next_temp_filename():
            return os.path.join(temp_dir, 'rht'+ str(len(temp_files)) + '.dat')
        print 'Temporary files in:', temp_dir 

        #Number of RHT operations that will be performed, and their coordinates
        update_progress(0.0, message)
        coords = zip( *np.nonzero( wlen_mask))
        N = len(coords)
        for c in range(N):
            j,i = coords[c]
            h = fast_hough(masked_udata[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1], xyt, ntheta)
            hout = np.divide(h, h1) #TODO____________________np.true_divide(h, h1)
            hout *= np.greater_equal(hout, frac)
            if np.any(hout):
                htapp(hout)
                hiapp(i)
                hjapp(j)
                if len(Hthets) == buffer_shape[0]:
                    temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=buffer_shape )) #Creates full memmap object
                    theta_array = np.array(Hthets, dtype=DTYPE) #Convert list to array
                    temp_files[-1][:] = theta_array[:] #Write array to last memmapped object in list
                    Hthets = [] #Reset Hthets
            update_progress((c+1)/float(N), message)
            #End

        if len(Hthets) > 0:
            temp_files.append( np.memmap( next_temp_filename(), dtype=DTYPE, mode='w+', shape=(len(Hthets), ntheta)  )) #Creates small memmap object
            theta_array = np.array(Hthets, dtype=DTYPE) #Convert list to array
            temp_files[-1][:] = theta_array[:] #Write array to last memmapped object in list

        #print 'Converting list of buffered hthet arrays into final XYT cube'
        converted_hthets = concat_along_axis_0(temp_files) #Combines memmap objects sequentially
        putXYT(xyt_filename, np.array(Hi), np.array(Hj), converted_hthets) #Saves data
        converted_hthets.flush()
        #converted_hthets.close() #Did not work?
        del converted_hthets

        
        try:
            for obj in os.listdir(temp_dir):
                obj.close()
                del obj
            shutil.rmtree(temp_dir) #Did not work!?
        except:
            print 'Failed to delete files in:', temp_dir
        finally:
            return True
            

#-----------------------------------------------------------------------------------------
#Interactive Functions
#-----------------------------------------------------------------------------------------

def rht(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR):
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
    if not is_valid_file(filepath):
        #Checks to see if a file should have the rht applied to it...
        print 'Invalid filepath encountered in rht('+filepath+')...'
        return False

    try:
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        output='.'
        xyt_filename = os.path.join(output, filename + '_xyt.npz')

        if not force and os.path.isfile(xyt_filename):
            return True
            #TODO CHECK WHETHER AN OUTPUT EXISTS WITH THIS PARAMETER INPUT FIRST!! ______________________________________________

        print '1/4:: Retrieving Data from:', filepath
        data, smr_mask, wlen_mask = getData(filepath, make_mask=True, smr=smr, wlen=wlen)
        datay, datax = data.shape

        print '2/4::', 'Size:', str(datax)+'x'+str(datay), 'Wlen:', str(wlen)+',', 'Smr:', str(smr)+',', 'Frac:', str(frac)
        
        ntheta = ntheta_w(wlen) #TODO_______________ntheta
        theta, dtheta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False, retstep=True) #____________________________________________________________________________________________ 2*       

        message = '3/4:: Running RHT...'
        success = window_step(data, wlen, frac, smr, theta, ntheta, smr_mask, wlen_mask, xyt_filename, message) #TODO__________________
        

        print '4/4:: Successfully Saved Data As', xyt_filename
        return success
    
    except:
        raise #__________________________________________________________________________________________________________ Raise
        return False


def interpret(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR):
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
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    xyt_filename = filename + '_xyt.npz'
    required_files = [xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing:
        #Runs rht(filepath), since that has not been done
        rht(filepath, force=force, wlen=wlen, frac=frac, smr=smr) 
    else:
        if force:
            #Runs rht(filepath), even if it has been done
            rht(filepath, force=True, wlen=wlen, frac=frac, smr=smr)
        else:
            #Good to go! No rht needed!
            pass 


    #Proceed with iterpreting the rht output files
    hi, hj, hthets = getXYT(xyt_filename, rebuild=False)

    #Spectrum *Length ntheta array of theta power (above the threshold) for whole data*
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

    
def viewer(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR):
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
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    xyt_filename = filename + '_xyt.npz'
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'
    required_files = [backproj_filename, spectrum_filename, xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing:
        #Interprets file, since that has not been done
        interpret(filepath, force=force, wlen=wlen, frac=frac, smr=smr) 
    else:
        if force:
            #Interprets file, even if it has been done
            interpret(filepath, force=True, wlen=wlen, frac=frac, smr=smr)
        else:
            #Good to go! No interpretation needed!
            pass 

    #Loads in relevant files and data___________________________________
    #data, smr_mask, wlen_mask = getData(filepath, make_mask=True, smr=smr, wlen=wlen)
    data = getData(filepath)
    backproj = np.load(backproj_filename).astype(np.float)
    spectrum = np.load(spectrum_filename).astype(np.float)
    hi, hj, hthets = getXYT(xyt_filename)

    #Gather parameters from that data
    ntheta = len(spectrum)
    datay, datax = data.shape

    #Produce Specific Plots
    masked_udata = umask(data, smr)
    log = np.log(np.where( np.isfinite(data), data, np.ones_like( data) )) #TODO Warnings?
    #U = np.zeros_like(hi)
    #V = np.zeros_like(hj)
    #C = np.zeros((len(U)), dtype=np.float)
    #coords = zip(hi, hj)
    #for c in range(len(coords)):
        #C[c], U[c], V[c] = theta_rht(hthets[c], uv=True)
    #C /= np.pi 
    #C *= np.isfinite(C)
    #np.clip(C, 0.0, 1.0, out=C)

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
    #error = 0.0001
    #reduction = np.nonzero( [0.0+error <c< 0.5-error or 0.5+error <c< 1.0-error for c in C] )[::4]
    #axes[1][0].quiver(hi[reduction], hj[reduction], U[reduction], V[reduction], C[reduction], units='xy', pivot='middle', scale=1.0, cmap='hsv')
    #axes[1][0].set_ylim([0, datay])
    #axes[1][0].set_title('Theta<RHT>')

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
    
    
    #---------------------------------------------------------------------- PLOT 3
    print 'Theta Spectrum'

    plt.plot(np.linspace(0.0, 2*180.0, num=ntheta, endpoint=False), spectrum) #___________________________________________________________________________---2*
    #___________________________________TODO SAVE PLOT
    plt.show()
    cleanup()
    
    #---------------------------------------------------------------------- PLOT 4
    #Polar plot of theta power
    print 'Linearity'
    plt.polar(np.linspace(0.0, 2*np.pi, num=ntheta, endpoint=False), spectrum)
    plt.show()
    cleanup()
    

    '''
    #---------------------------------------------------------------------- PLOT 4
    n = 5
    steps = np.linspace(frac, 1.0, n, endpoint=True, retstep=False)
    #cmap = 'hot'
    #from  matplotlib.colors import BoundaryNorm
    #norm = BoundaryNorm(steps, cmap.N, clip=True)

    xyt_filename = filename + '_xyt.npz'
    #hi, hj, hthets = getXYT(xyt_filename, rebuild=False)
    xyt = getXYT(xyt_filename, rebuild=True, filepath=filepath)
    datay, datax, ntheta = xyt.shape
    
    for i in range(n-1):
        low, high = steps[i], steps[i+1]
        boolean = np.logical_and( np.greater_equal(xyt, low), np.less(xyt, high) )
        (ys, xs, zs)  = np.nonzero(boolean)
        #vectors = transpose((xs, ys, zs))
        #s = 20*xyt[ys, xs,zs]

        z_arr = np.zeros(ntheta)
        for a in range(datay):
            y = np.nonzero(np.equal(ys,a))
            for b in range(datax):
                x = np.nonzero(np.equal(xs,b))
                j = []
                for index in y:
                    if index in x:
                        j.append(index)
                if len(j) > 0:
                    j = np.array(j).astype(np.int)
                    z_arr[-1, zs[j]] = xyt[a, b, zs[j]]
                    z_arr = np.append(z_arr, np.zeros(ntheta), axis=0)

        axes[1][0].scatter(xs, ys, z_arr) #cmap=cmap, norm=norm)
        
    axes[1][0].set_ylim([0, datay-1], auto=True)
    axes[1][0].set_xlim([0, datax-1], auto=True)
    axes[1][0].set_zlim([0, ntheta-1], auto=True)
    axes[1][0].set_title('3D Scatterplot of ThetaRHT')
    '''

    #---------------------------------------------------------------------- PLOTS END
    #Clear all and exit successfully
    cleanup()
    return True


def main(source=None, display=False, force=False, wlen=WLEN, frac=FRAC, smr=SMR):
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
        #print 'File Detected'
        pathlist.append(source)
    elif os.path.isdir(source):
        #Input = Directory
        #print 'Directory Detected'
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
    #TODO batch progress bar

    if not 0 <= frac <= 1:
        frac = FRAC
    if not (wlen > 0 and wlen%2):
        wlen = WLEN
    if not (smr > 0 and smr%2):
        smr = SMR

    summary = []
    for path in pathlist:
        success = True
        try:
            if (display):
                success = viewer(path, force=force, wlen=wlen, frac=frac, smr=smr)
            else:
                success = rht(path, force=force, wlen=wlen, frac=frac, smr=smr)
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

#TODO
        
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
  -f  #Exisitng _xyt.npz is to be Forcefully overwritten

  Params:
  -wlen=value  #Sets window diameter
  -smr=value  #Sets smoothing radius
  -frac=value  #Sets theta power threshold'''
    
    if len(sys.argv) == 1:
        #Displays the README file   
        README = 'README'
        try:
            readme = open(README, 'r')
            print readme.read(2000) #TODO
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
                argval = arg.lower().split('=')[1] #TODO Handle errors
                if argname in ['w', 'wlen', '-w', '-wlen']:
                    wlen = float(argval)
                elif argname in ['s', 'smr', '-s', '-smr']:
                    smr = float(argval)
                elif argname in ['f', 'frac', '-f', '-frac']:
                    frac = float(argval)
                else:
                    print 'UNKNOWN PARAMETER:', arg

        main(source=SOURCE, display=DISPLAY, force=FORCE, wlen=wlen, frac=frac, smr=smr)

    exit()

#-----------------------------------------------------------------------------------------
#Attribution
#-----------------------------------------------------------------------------------------

#This is the Rolling Hough Transform, described in Clark, Peek, Putman 2014 (arXiv:1312.1338).
#Modifications to the RHT have been made by Lowell Schudel, CC'16.
