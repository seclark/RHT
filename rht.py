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
FRAC = 0.75 #0.70 #fraction (percent) of one angle that must be 'lit up' to be counted
SMR = 5 #11.0 #smoothing radius of unsharp mask function
# ulen : length of unsharp mask square. Must be at least wlen + smr/2

#-----------------------------------------------------------------------------------------
#Initialization
#-----------------------------------------------------------------------------------------
'''

'''

#-----------------------------------------------------------------------------------------
#Utility Functions
#-----------------------------------------------------------------------------------------
def is_valid_file(filepath):
    '''
    filepath: Potentially a string path to a source image

    return: Boolean, True ONLY when the image could have rht() applied successfully
    '''
    excluded_file_endings = ['_xyt.npz', '_backproj.npy', '_spectrum.npy'] #TODO___More Endings
    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False
    
    excluded_file_content = ['_xyt', '_backproj', '_spectrum'] #TODO___More Exclusions
    if any([e in filepath for e in excluded_file_content]):
        return False

    return True

def center(filepath, shape=(500, 500)):
    #Returns a cutout from the center of the image
    xy_array = getData(filepath)
    datay, datax = xy_array.shapy 
    x, y = shape
    if 0 < x < datax and 0 < y < datay:
        left = int(datax//2-x//2)
        right = int(datax//2+x//2)
        up = int(datay//2+y//2)
        down = int(datay//2-y//2)
        cutout = np.array(xy_array[down:up, left:right])
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        center_filename = filename+'_center.npy'
        np.save(center_filename, cutout)
        return center_filename
    else:
        return None 

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

def putXYT(xyt_filename, hi, hj, hthets, compressed=True):
    if compressed:
        np.savez_compressed(xyt_filename, hi=hi, hj=hj, hthets=hthets)
    else:
        np.savez(xyt_filename, hi=hi, hj=hj, hthets=hthets)

def getXYT(xyt_filename, rebuild=False):    
    #Reads in a .npz file containing coordinate pairs in image space (hi, hj)
    #And Hough space arrays covering theta space at each of those points
    data = np.load(xyt_filename)
    hi = data['hi']
    hj = data['hj']
    hthets = data['hthets']
    if rebuild:
        #Can recreate an entire 3D array of mostly 0s
        image = getData(filepath)
        imy, imx = image.shape
        ntheta = len(hthets[0])
        xyt = np.zeros((imy, imx, ntheta))
        coords = zip(hi, hj)
        for c in range(len(coords)):
            xyt[coords[c][1]][coords[c][0]] = hthets[c]
        return xyt
    else:
        #Returns the sparse form only
        return hi, hj, hthets

def getData(filepath, make_mask=False, wlen=WLEN):
    #Reads in and makes proper masks for images from various sources
    #Supports .fits, .npy, and PIL formats
    print 'Retrieving Data'
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
            print 'Failure in getData('+filepath+')... Returning None, None'
            return None, None
        else:
            print 'Failure in getData('+filepath+')... Returning None'
            return None 

    #TODO___________________________________________________________________________clean_data
    #Sets all non-finite values (NaN or Inf) to the minimum finite value of data
    #datamin = np.nanmin(data)
    #clean_data = np.where(np.isfinite(data), data, datamin*np.ones_like(data))
    
    #Sets all non-finite values (NaN or Inf) to zero
    clean_data = np.where(np.isfinite(data), data, np.zeros_like(data))

    if not make_mask:
        #Done Reading Data, No Mask Needed
        return clean_data
    else:
        #Mask Needed
        update_progress(0.0, message='Masking::')
        wsquare1 = np.ones((wlen, wlen), np.int_) #TODO ____________INCLUDE SMR HERE
        wkernel = circ_kern(wsquare1, wlen)
        try:
            isZEA = filepath.endswith('.fits') and any(['ZEA' in x.upper() for x in [hdu.header['CTYPE1'], hdu.header['CTYPE2']] ])
        except:
            #CTYPE1 or CTYPE2 not found in header keys
            #Assume file must not be ZEA... Treat as a rectangle
            isZEA = False
        finally:
            if isZEA:
                #Making Round Mask for ZEA data #TODO header values__________________________________________________
                #mask = makemask(wkernel, data)
                # The following is specific to a certain data set (the Parkes Galactic All-Sky Survey)
                # which was in a Zenith-Equal-Area projection. This projects the sky onto a circle, and so 
                # makemask just makes sure that nothing outside that circle is counted as data.
                
                datay, datax = data.shape
                
                mnvals = np.indices(data.shape)
                pixcrd = np.zeros((datax*datay,2), np.float_)
                pixcrd[:,0] = mnvals[:,:][0].reshape(datax*datay)
                pixcrd[:,1] = mnvals[:,:][1].reshape(datax*datay)
                
                w = wcs.WCS(naxis=2)
                #TODO READ FROM FITS HEADER FILE!
                w.wcs.crpix = [1.125000000E3, 1.125000000E3]
                w.wcs.cdelt = np.array([-8.00000000E-2, 8.00000000E-2])
                w.wcs.crval = [0.00000000E0, -9.00000000E1]
                w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA']
                
                worldc = w.wcs_pix2world(pixcrd, 1)
                worldcra = worldc[:,0].reshape(*data.shape)
                worldcdec = worldc[:,1].reshape(*data.shape)
                
                gm = np.zeros_like(data)
                gmconv = scipy.ndimage.filters.correlate(gm, weights=wkernel)
                
                gg = gmconv.copy() #copy.copy(gmconv)
                gg[gmconv < np.max(gmconv)] = 0
                gg[gmconv == np.max(gmconv)] = 1
                
                mask = gg

            else:
                #Mask a Rectangular chunk of a rectangular image!
                mask = np.zeros_like(data)
                wcntr = int(np.floor(wlen/2)) #TODO______________________________ Indexing?
                datay, datax = data.shape
                mask[wcntr:datay-wcntr, wcntr:datax-wcntr] = 1 #TODO______________________________ Indexing?
                #Mask any pixel within wcntr of a NaN pixel
                y_arr, x_arr = np.nonzero(wkernel)
                '''
                for (j,i) in zip(*np.nonzero(mask)):
                '''
                coords = zip(*np.nonzero(mask))
                for c in range(len(coords)):
                    j,i = coords[c]

                    x = x_arr - wcntr + i
                    y = y_arr - wcntr + j
                    a =  np.isfinite( data[y.astype(np.int), x.astype(np.int)] ).all()
                    mask[j][i] = a
                    update_progress(c/(len(coords)-1), message='Masking::')

            return clean_data, mask 

def setParams(w, s, f):
    wlen = float(w)  #Window diameter
    frac = float(f) #Theta-power threshold to store
    smr = float(s) #Smoothing radius

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



def makemask(wkernel, data):
    # The following is specific to a certain data set (the Parkes Galactic All-Sky Survey)
    # which was in a Zenith-Equal-Area projection. This projects the sky onto a circle, and so 
    # makemask just makes sure that nothing outside that circle is counted as data.
    
    datay, datax = data.shape
    
    mnvals = np.indices(data.shape)
    pixcrd = np.zeros((datax*datay,2), np.float_)
    pixcrd[:,0] = mnvals[:,:][0].reshape(datax*datay)
    pixcrd[:,1] = mnvals[:,:][1].reshape(datax*datay)
    
    w = wcs.WCS(naxis=2)
    #TODO READ FROM FITS HEADER FILE!
    w.wcs.crpix = [1.125000000E3, 1.125000000E3]
    w.wcs.cdelt = np.array([-8.00000000E-2, 8.00000000E-2])
    w.wcs.crval = [0.00000000E0, -9.00000000E1]
    w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA']
    
    worldc = w.wcs_pix2world(pixcrd, 1)
    worldcra = worldc[:,0].reshape(*data.shape)
    worldcdec = worldc[:,1].reshape(*data.shape)
    
    gm = np.zeros_like(data)
    gmconv = scipy.ndimage.filters.correlate(gm, weights=wkernel)
    
    gg = gmconv.copy() #copy.copy(gmconv)
    gg[gmconv < np.max(gmconv)] = 0
    gg[gmconv == np.max(gmconv)] = 1
    
    return gg


#Performs a circle-cut of given radius on inkernel.
#Outkernel is 0 anywhere outside the window.    
def circ_kern(inkernel, radius):
    #These are all the possible (m,n) indices in the image space, centered on center pixel
    mnvals = np.indices(inkernel.shape)
    kcntr = np.floor(len(inkernel)/2.0)
    mvals = mnvals[:,:][0] - kcntr
    nvals = mnvals[:,:][1] - kcntr

    rads = np.sqrt(nvals**2 + mvals**2)
    outkernel = inkernel.copy() #copy.copy(inkernel)
    outkernel[rads > radius/2] = 0
    
    return outkernel

#Unsharp mask. Returns binary data.
def umask(data, inkernel):    
    outdata = scipy.ndimage.filters.correlate(data, weights=inkernel)
    
    #Our convolution has scaled outdata by sum(kernel), so we will divide out these weights.
    kernweight = np.sum(inkernel, axis=0)
    kernweight = np.sum(kernweight, axis=0)
    subtr_data = data - outdata/kernweight
    
    #Convert to binary data
    bindata = subtr_data.copy() #copy.copy(subtr_data)
    bindata[subtr_data > 0] = 1
    bindata[subtr_data <= 0] = 0

    return bindata

def fast_hough(in_arr, xyt, ntheta):
    incube = np.repeat(in_arr[:,:,np.newaxis], repeats=ntheta, axis=2)
    out = np.sum(np.sum(incube*xyt,axis=0), axis=0)
    
    return out        

def all_thetas(window, thetbins):
    wy, wx = window.shape #Parse x/y dimensions
    ntheta = len(thetbins) #Parse height in theta
    
    #Makes prism; output has dimensions (x, y, theta)
    out = np.zeros((wy, wx, ntheta), np.int_)
    
    for i in xrange(wx):
        for j in xrange(wy):
            #At each x/y value, create new single-pixel image
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
        raise ValueError('The input image must be 2-D')

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
        #out.T[i] = bincount

    return out, theta, bins


def window_step(data, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask): 
    update_progress(0.0)
    
    #Circular kernels
    wsquare1 = np.ones((wlen, wlen), np.int_) #Square of 1s
    kernel = circ_kern(wsquare1, smr) #Stores an smr-sized circle
    wkernel = circ_kern(wsquare1, wlen) #And an wlen-sized circle
    xyt = all_thetas(wkernel, theta) #Cylinder of all theta values per point

    #unsharp mask the whole data set
    udata = umask(data, kernel)

    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(wkernel, xyt, ntheta) #Length ntheta array
    dcube = np.repeat(udata[:,:,np.newaxis], repeats=ntheta, axis=2)

    Hthets = []
    Hi = []
    Hj = []
    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    npsum = np.sum
    '''
    #Loop: (j,i) are centerpoints of data window.
    datay, datax = data.shape 
    #TODO-------------------------------------------------------
    ucntr = int(ucntr) #should already be one
    for j in xrange(ucntr, (datay - ucntr)):         
        update_progress((j-ucntr)/(datay-2.0*ucntr-1.0)) 
        for i in xrange(ucntr, (datax - ucntr)): 
    


            for j in xrange(datay): 
                if j >= ucntr and j < (datay - ucntr):        
                    update_progress((j-ucntr)/(datay-2.0*ucntr-1.0)) 
                    for i in xrange(datax): 
                        if i >= ucntr and i < (datax - ucntr):
    
            
     
            start = ucntr
            stopy = (datay-ucntr)
            stopx = (datax-ucntr)
            if stopx == np.floor(stopx): #TODO this is purely < stop
                stopx += 1 
            if stopy == np.floor(stopy): 
                stopy += 1
            for j in np.arange(start, stopy, 1):        
                update_progress((j-start)/(stopy-start-1.0))
                for i in np.arange(start, stopx, 1):
    
    
            #TODO-------------------------------------------------------
            if mask is None or (mask is not None and mask[j,i] == 1):
    '''
    coords = zip(*np.nonzero(mask))
    for c in range(len(coords)):
        j,i = coords[c]
        update_progress(c/(len(coords)-1), '3/4::')
        try:
            wcube = dcube[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1, :]   
            h = npsum(npsum(wcube*xyt,axis=0), axis=0) 
            hout = h/h1 - frac #h, h1 are Length ntheta arrays and frac is a float
            hout[hout<0.0] = 0.0
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
    filepath: String path to source image, which will have the Rolling Hough Transform applied
    force: Boolean indicating if rht() should still be run, even when output exists for these inputs

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an image

    Saves:
        X-Y-ThetaPower Array --> name_xyt.npz

    return: Boolean, if the function succeeded
    '''
    if not is_valid_file(filepath):
        #Checks to see if a file should have the rht applied to it...
        print 'Invalid filepath in rht('+filepath+')...'
        return False

    try:
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        output='.'
        xyt_filename = os.path.join(output, filename + '_xyt.npz')

        if not force and os.path.isfile(xyt_filename):
            return True
            #TODO CHECK WHETHER AN OUTPUT EXISTS WITH THIS PARAMETER INPUT FIRST!! ______________________________________________

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
    except:
        raise #___________________________________________________________________________
        return False


def interpret(filepath, force=False, wlen=WLEN, frac=FRAC, smr=SMR):
    '''
    filepath: String path to source image, used in forcing and backprojection
    force: Boolean indicating if rht() should be run, even when required_files are found

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an image

    Saves:
        Backprojection --> name_backproj.npy
        ThetaSpectrum --> name_spectrum.npy

    return: Boolean, if the function succeeded
    '''
    #Read in rht output files
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    xyt_filename = filename + '_xyt.npz'

    #Makes sure relevant files are present! 
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


    #Proceed with iterpreting
    hi, hj, hthets = getXYT(xyt_filename, rebuild=False)

    #Spectrum *Length ntheta array of theta power (above the threshold) for whole image*
    spectrum = [np.sum(theta) for theta in hthets]
    spectrum_filename = filename + '_spectrum.npy'
    np.save(spectrum_filename, np.array(spectrum))

    #Backprojection only *Full Size* #Requires image
    image = getData(filepath)
    imy, imx = image.shape
    backproj = np.zeros_like(image)
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
    filepath: String path to source image, used in forcing and backprojection
    force: Boolean indicating if interpret() should be run, even when required_files are found

    wlen: Diameter of a 'window' to be evaluated at one time
    frac: Fraction in [0.0, 1.0] of pixels along one angle that must be 'lit up' to be counted
    smr: Integer radius of gaussian smoothing kernel to be applied to an image
    
    Plots:
        name_backproj.npy --> Backprojection 
        name_spectrum.npy --> ThetaSpectrum, Linearity

    return: Boolean, if the function succeeded
    '''
    #Loads in relevant files
    image, mask = getData(filepath, make_mask=True, wlen=wlen)
    imy, imx = image.shape
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'

    #Makes sure relevant files are present!
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
    
    def cleanup(all=False):
        plt.clf()
        plt.cla()
        if all:
            plt.close('all')
        else:
            plt.close()

    print 'Whole Figure'
    fig, axes = plt.subplots(nrows=2, ncols=2)
    c = 0.1
    a = np.log(np.abs(np.where(mask, image, c*np.ones_like(image)))) 
    axes[0][0].imshow(a, cmap='gray')
    axes[0][0].set_ylim([0, imy])
    axes[0][0].set_title(filepath)

    axes[1][1].imshow(np.load(backproj_filename), cmap='gray') #cmap='binary')
    axes[1][1].set_ylim([0, imy])
    axes[1][1].set_title(backproj_filename)

    axes[1][0].imshow(mask, cmap='gray') #cmap='binary')
    axes[1][0].set_ylim([0, imy])
    axes[1][0].set_title('Mask')

    udata = umask(image, circ_kern(np.ones((wlen, wlen), np.int_), smr))
    axes[0][1].imshow(udata, cmap='gray') #cmap='binary')
    axes[0][1].set_ylim([0, imy])
    axes[0][1].set_title('Sharpened')

    plt.show(fig)
    #___________________________________TODO SAVE PLOT
    del a, c, udata
    cleanup()

    '''
    print 'Backprojection'
    #contour(np.load(backproj_filename))
    plt.subplot(121)
    c = 0.1
    a = np.log(np.abs(np.where(np.logical_and(image, np.isfinite(image)), image, c*np.ones_like(image)))) 
    plt.imshow(a, cmap='gray')
    plt.ylim(0, imy)
    plt.title(filepath)
    plt.subplot(122)
    plt.imshow(np.load(backproj_filename), cmap='binary')
    plt.ylim(0, imy)
    plt.title(backproj_filename)
    plt.show()
    #___________________________________TODO SAVE PLOT
    del a, c
    cleanup()
    '''

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
    smr: Integer radius of gaussian smoothing kernel to be applied to an image

    return: Boolean, if the function succeeded
    '''
    #Ensure that the input is a non-None string
    while source is None or type(source) != str: #TODO Fix escape char bug
        try:
            source = raw_input('Please enter the name of a file or directory to transform:')
        except:
            source = None
    
    #______________________________________________________________________________CLEANS SOURCE
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
 Creates 'dirname/filename_xyt.npz' for each input image
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
