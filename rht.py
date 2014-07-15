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


#-----------------------------------------------------------------------------------------
#Default Parameters
#-----------------------------------------------------------------------------------------
TEXTWIDTH = 60 #Width of some displayed text objects
WLEN = 55 #101.0 #Diameter of a 'window' to be evaluated at one time
FRAC = 0.70 #0.70 #fraction (percent) of one angle that must be 'lit up' to be counted
SMR = 11.0 #smoothing radius of unsharp mask function

#-----------------------------------------------------------------------------------------
#Initialization
#-----------------------------------------------------------------------------------------
'''

'''

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
    #Create progress meter
    length = TEXTWIDTH//2
    messlen = TEXTWIDTH-(length+4)
    message = string.ljust(message, messlen)[:messlen]
    if 0.0 <= progress <= 1.0:
        p = int(length*progress/1.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(length-p), message))
        sys.stdout.flush()
        if p == length:
            print ''
    elif 0.0 <= progress <= 100.0:
        p = int(length*progress/100.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(length-p), message)) 
        sys.stdout.flush()
        if p == length:
            print ''
    else:
        pass ##TODO Progress Bar Failure

#-----------------------------------------------------------------------------------------
#Image Processing Functions
#-----------------------------------------------------------------------------------------
def is_valid_file(filepath):
    '''
    filepath: Potentially a string path to a source data

    return: Boolean, True ONLY when the data could have rht() applied successfully
    '''
    excluded_file_endings = ['_xyt.npz', '_backproj.npy', '_spectrum.npy'] #TODO___More Endings
    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False
    
    excluded_file_content = ['_xyt', '_backproj', '_spectrum'] #TODO___More Exclusions
    if any([e in filepath for e in excluded_file_content]):
        return False

    return True

def center(filepath, shape=(500, 500)):
    #Returns a cutout from the center of the data
    data = getData(filepath)
    datay, datax = data.shape 
    x, y = shape
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
        return None 

def putXYT(xyt_filename, hi, hj, hthets, compressed=True):
    if compressed:
        np.savez_compressed(xyt_filename, hi=hi, hj=hj, hthets=hthets)
    else:
        np.savez(xyt_filename, hi=hi, hj=hj, hthets=hthets)

def getXYT(xyt_filename, rebuild=False):    
    #Reads in a .npz file containing coordinate pairs in data space (hi, hj)
    #And Hough space arrays covering theta space at each of those points
    data = np.load(str(xyt_filename))
    hi = data['hi']
    hj = data['hj']
    hthets = data['hthets']
    if rebuild:
        #Can recreate an entire 3D array of mostly 0s
        data = getData(filepath)
        imy, imx = data.shape
        ntheta = len(hthets[0])
        xyt = np.zeros((imy, imx, ntheta))
        coords = zip(hj, hi)
        for c in range(len(coords)):
            j,i = coords[c]
            xyt[j][i] = hthets[c]
        return xyt
    else:
        #Returns the sparse form only
        return hi, hj, hthets

def bad_pixels(data):
    #Returns an array of the same shape as data
    #Bad values become 1, all else become 0
    #return np.isnan(data)
    data = np.array(data, np.float)
    return np.logical_not(np.nan_to_num(data)) #(Nans or 0) ---> (0) ---> (1)

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
            hdu = fits.open(filepath)[0] #Opens first HDU
            data = hdu.data #Reads all data as an array

        elif filepath.endswith('.npy'):
            data = np.load(filepath) #Reads numpy files #TODO
        
        else:
            data = scipy.ndimage.imread(filepath, True)[::-1] #Makes B/W array, reversing y-coords         
    except:
        #Failure Reading Data
        if make_mask:
            print 'Failure in getData('+filepath+')... Returning None, None, None'
            return None, None, None
        else:
            print 'Failure in getData('+filepath+')... Returning None'
            return None 

    #TODO___________________________________________________________________________clean_data
    #Sets all non-finite values (NaN or Inf) to the minimum finite value of data
    #datamin = np.nanmin(data)
    #clean_data = np.where(np.isfinite(data), data, datamin*np.ones_like(data))
    #Sets all non-finite values (NaN or Inf) to zero
    #clean_data = np.where(np.isfinite(data), data, np.zeros_like(data))

    if not make_mask:
        #Done Reading Data, No Mask Needed
        return data
    else:
        #Mask Needed, cuts away smr radius from bads, then wlen from bads 
        smr_mask = all_within_diameter_are_good(data, 2*smr)
        nans = np.empty(data.shape, dtype=np.int).fill(np.nan)
        wlen_mask = all_within_diameter_are_good( np.where(smr_mask, data, nans), wlen)

        datamin = np.nanmin(data)
        datamax = np.nanmax(data)
        datamean = np.nanmean(data)
        print 'Data File Info, Min:', datamin, 'Mean:', datamean, 'Max:', datamax

        return data, smr_mask, wlen_mask

#Performs a circle-cut of given diameter on inkernel.
#Outkernel is 0 anywhere outside the window.   
def circ_kern(diameter):
    r = int(np.floor(diameter/2))
    mnvals = np.indices((diameter, diameter)) - r
    rads = np.hypot(mnvals[0], mnvals[1])
    return np.less_equal(rads, r).astype(np.int)

#Unsharp mask. Returns binary data.
def umask(data, radius, smr_mask=None):
    kernel = circ_kern(2*radius)
    outdata = scipy.ndimage.filters.correlate(data, weights=kernel)
    
    #Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(np.sum(kernel, axis=0), axis=0)
    subtr_data = data - outdata/kernweight
    
    #Convert to binary data
    bindata = np.greater(subtr_data, 0)

    if smr_mask == None:
        return bindata
    else:
        #nans = np.empty(bindata.shape, dtype=np.int)
        #nans.fill(np.nan)
        #return np.where(smr_mask, bindata, nans) #TODO________________________________________May Not be right?
        return np.where(smr_mask, bindata, smr_mask) 

def fast_hough(in_arr, xyt, ntheta):
    incube = np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)
    out = np.sum(np.sum(incube*xyt,axis=0), axis=0)
    
    return out        

def all_thetas(window, thetbins):
    wy, wx = window.shape #Parse x/y dimensions
    ntheta = len(thetbins) #Parse height in theta
    
    #Makes prism; output has dimensions (x, y, theta)
    out = np.zeros((wy, wx, ntheta), np.int)
    
    for i in xrange(wx):
        for j in xrange(wy):
            #At each x/y value, create new single-pixel data
            w_1 = np.zeros((wy, wx), np.float_)
            
            # run the Hough for each point one at a time
            if window[j,i] == 1:
                w_1[j,i] = 1
       
                H, thets, dist = houghnew(w_1, thetbins) 
                rel = H[np.floor(len(dist)/2), :] 
                out[j, i, :] = rel
      
    return out    

def houghnew(img, theta=None, idl=False):
    if img.ndim != 2:
        raise ValueError('The input data must be 2-D')

    if theta is None:
        theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, 180)
    
    wy, wx = img.shape 
    wmid = np.floor(wx/2.0)
    
    if idl:
        ntheta = math.ceil((np.pi*np.sqrt(2.0)*((wx-1)/2.0)))  
        theta = np.linspace(0, np.pi, ntheta)

    # compute the vertical bins (the distances)
    d = np.ceil(np.hypot(*img.shape))
    nr_bins = d
    bins = np.linspace(-d/2, d/2, nr_bins)

    # allocate the output data
    out = np.zeros((nr_bins, len(theta)), dtype=np.uint64)

    # precompute the sin and cos of the angles
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # find the indices of the non-zero values in
    # the input data
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
        #out.T[i] = bincount

    return out, theta, bins


def window_step(data, wlen, frac, smr, theta, ntheta, smr_mask, wlen_mask): 
    update_progress(0.0, '3/4::')
    wcntr = int(np.floor(wlen/2))

    #Circular kernels
    wkernel = circ_kern(wlen)
    xyt = all_thetas(wkernel, theta) #Cylinder of all theta values per point

    #Unsharp masks the whole data set
    masked_udata = umask(data, smr, smr_mask=smr_mask) 

    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(wkernel, xyt, ntheta) #Length ntheta array
    dcube = np.repeat(masked_udata[:,:,np.newaxis], repeats=ntheta, axis=2)

    Hthets = []
    Hi = []
    Hj = []
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    npsum = np.sum

    coords = zip( *np.nonzero( np.logical_and( wlen_mask, smr_mask )))
    for c in range(len(coords)):
        j,i = coords[c]
        update_progress(c/(len(coords)-1), '3/4::')
        try:
            wcube = dcube[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1, :]   
            h = npsum(npsum(wcube*xyt,axis=0), axis=0) 
            #________________________________________
            #hout = h/h1 - frac #h, h1 are Length ntheta arrays and frac is a float
            #hout[hout<0.0] = 0.0
            #_____________________________________________________________________________________________TODO
            hout = np.divide(h, h1) #np.true_divide(h, h1)
            hout *= np.greater_equal(hout, frac)
            #________________________________________
            if any(hout):
                htapp(hout)
                hiapp(i)
                hjapp(j)
        except:
            print 'Failure:', i, j, wcntr
            raise
        #finally:
            #pass 

    return np.array(Hthets), np.array(Hi), np.array(Hj)

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
        
        ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0))) #TODO_______________ntheta
        theta, dtheta = np.linspace(0.0, np.pi, ntheta, endpoint=False, retstep=True)        

        hthets, hi, hj = window_step(data, wlen, frac, smr, theta, ntheta, smr_mask, wlen_mask) #TODO__________________
        

        putXYT(xyt_filename, hi, hj, hthets)

        print '4/4:: Successfully Saved Data As', xyt_filename
        return True
    
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
        rht(filepath, force=force, wlen=wlen, frac=frac, smr=smr)       #TODO FORCE__________
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
    spectrum = [np.sum(theta) for theta in hthets] #TODO_____________________________________Counts instead of Sum?? #np.count_nonzero(theta)
    spectrum_filename = filename + '_spectrum.npy'
    np.save(spectrum_filename, np.array(spectrum))

    #Backprojection only *Full Size* #Requires data
    data = getData(filepath)
    imy, imx = data.shape
    backproj = np.zeros_like(data)
    coords = zip(hi, hj)
    for c in range(len(coords)):
        backproj[coords[c][1]][coords[c][0]] = np.sum(hthets[c]) 
    #for c in coords: #SLOW VERSION, EQUIVALENT TO ABOVE
        #backproj[c[1]][c[0]] = np.sum(hthets[coords.index(c)]) 
    np.divide(backproj, np.sum(backproj), backproj)
    backproj_filename = filename + '_backproj.npy'
    np.save(backproj_filename, np.array(backproj))

    print 'Success'
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
    #Makes sure relevant files are present!
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'
    required_files = [backproj_filename, spectrum_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing:
        #Interprets file, since that has not been done
        interpret(filepath, force=force, wlen=wlen, frac=frac, smr=smr)  #TODO FORCE__________
    else:
        if force:
            #Interprets file, even if it has been done
            interpret(filepath, force=True, wlen=wlen, frac=frac, smr=smr)
        else:
            #Good to go! No interpretation needed!
            pass 

    #Loads in relevant files
    data, smr_mask, wlen_mask = getData(filepath, make_mask=True, smr=smr, wlen=wlen)
    imy, imx = data.shape
    backproj = np.load(backproj_filename).astype(np.float)
    
    def cleanup(all=False):
        plt.clf()
        plt.cla()
        if all:
            plt.close('all')
        else:
            plt.close()

    print 'Whole Figure'
    fig, axes = plt.subplots(nrows=2, ncols=2)
    log = np.where( np.isfinite(data), data, np.ones_like( data) ) 
    axes[0][0].imshow(log, cmap='gray')
    axes[0][0].set_ylim([0, imy])
    axes[0][0].set_title(filepath)

    #print 'Backproj free of bad values:', np.all(np.isfinite(backproj))
    #back = np.where(wlen_mask, backproj, np.zeros_like(wlen_mask))
    axes[1][1].imshow(backproj, cmap='gray') #cmap='binary')
    axes[1][1].set_ylim([0, imy])
    axes[1][1].set_title(backproj_filename)

    #axes[1][0].imshow(np.logical_and(smr_mask, wlen_mask), cmap='gray') #cmap='binary')
    #axes[1][0].set_ylim([0, imy])
    #axes[1][0].set_title('Mask')

    masked_udata = umask(data, smr, smr_mask=smr_mask)
    axes[0][1].imshow(masked_udata, cmap='gray') #cmap='binary')
    axes[0][1].set_ylim([0, imy])
    axes[0][1].set_title('Sharpened')

    plt.show(fig)
    #___________________________________TODO SAVE PLOT
    del log, masked_udata
    cleanup()
    
    print 'Backprojection'
    plt.subplot(121)
    log = np.where( np.isfinite(data), data, np.ones_like( data) ) 
    plt.imshow(log, cmap='gray')
    plt.ylim(0, imy)
    plt.title(filepath)
    plt.subplot(122)
    #plt.contour(backproj)
    plt.imshow(backproj, cmap='binary')
    plt.ylim(0, imy)
    plt.title(backproj_filename)
    plt.show()
    #___________________________________TODO SAVE PLOT
    del log
    cleanup()
    
    print 'Theta Spectrum'
    spectrum = np.load(spectrum_filename)
    ntheta = len(spectrum)
    plt.plot(np.linspace(0.0, 180.0, num=ntheta, endpoint=False), spectrum)
    #___________________________________TODO SAVE PLOT
    plt.show()
    cleanup()

    #Polar plot of theta power
    print 'Linearity'
    r = np.append(spectrum, spectrum) / 2.0 
    t = np.linspace(0.0, 2*np.pi, num=len(r), endpoint=False)
    plt.polar(t, r)
    plt.show()
    cleanup()

    #Clear all and exit successfully
    cleanup(all=True)
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
        readme = open(README, 'r')
        print readme.read(2000) #TODO
        if len(readme.read(1)) == 1:
            print ''
            print '...see', README, 'for more information...'
            print ''
        readme.close()

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
