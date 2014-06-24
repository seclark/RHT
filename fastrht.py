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
from astropy import wcs
from astropy.io import fits
import scipy as sp
from scipy.ndimage import filters, imread
from matplotlib.pyplot import imshow, show
import copy
import sys

TEXTWIDTH = 40

def putXYT(xyt_filename, hi, hj, hthets, compressed=False):
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
        image, imx, imy = getData(filepath)
        xyt = np.zeros((imx, imy, len(hthets[0])))
        coords = zip(hi, hj)
        for c in range(len(coords)):
            xyt[coords[c][0]][coords[c][1]] = hthets[c]
        return xyt
    else:
        #Returns the sparse form only
        return hi, hj, hthets

def getData(filepath):
    #Reads in and properly rotates images from various sources
    #Supports .fits, .npy, and PIL formats
    if filepath.endswith('.fits'):
        hdulist = fits.open(filepath) #Opens HDU list
        gassslice = hdulist[0].data #Reads all data as an array
    elif filepath.endswith('.npy'):
        gassslice = np.load(filepath) #Reads numpy files
    else:
        gassslice = imread(filepath, True)[::-1] #Makes B/W array, reversing y-coords

    x, y = gassslice.shape #Gets dimensions
    return gassslice, x, y

def setParams(gassslice, w, s, f, ZEA=False):
    wlen = w #101.0 #Window diameter
    frac = f #0.70 #Theta-power threshold to store
    smr = s #11.0 #Smoothing radius

    ulen = np.ceil(wlen + smr/2) #Must be odd
    if np.mod(ulen, 2) == 0:
        ulen += 1
    ucntr = np.floor(ulen/2)

    wcntr = np.floor(wlen/2)
    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))  

    
    dtheta = np.pi/ntheta
    theta = np.arange(0, np.pi, dtheta)
    
    wsquare1 = np.ones((wlen, wlen), np.int_)
    kernel = circ_kern(wsquare1, smr) 
    wkernel = circ_kern(wsquare1, wlen) 
    
    if ZEA:
        mask = makemask(wkernel, gassslice)
    else:
        mask = None #Default is no mask

    return wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask


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
    #TODO READ FROM FITS HEADER FILE!
    w.wcs.crpix = [1.125000000E3, 1.125000000E3]
    w.wcs.cdelt = np.array([-8.00000000E-2, 8.00000000E-2])
    w.wcs.crval = [0.00000000E0, -9.00000000E1]
    w.wcs.ctype = ['RA---ZEA', 'DEC--ZEA']
    
    worldc = w.wcs_pix2world(pixcrd, 1)
    
    worldcra = worldc[:,0].reshape(datax,datay)
    worldcdec = worldc[:,1].reshape(datax,datay)
    
    gm = np.zeros(gassslice.shape)
    #gm[worldcdec < 0] = 1
    
    gmconv = filters.correlate(gm, weights=wkernel)
    gg = copy.copy(gmconv)
    gg[gmconv < np.max(gmconv)] = 0
    gg[gmconv == np.max(gmconv)] = 1
    
    return gg


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

#Unsharp mask. Returns binary data.
def umask(data, inkernel):    
    outdata = filters.correlate(data, weights=inkernel)
    
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


def window_step(data, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask): 

                    

    update_progress(0.0)
    
    #Circular kernels
    wsquare1 = np.ones((wlen, wlen), np.int_) #Square of 1s
    kernel = circ_kern(wsquare1, smr) #Stores an smr-sized circle
    wkernel = circ_kern(wsquare1, wlen) #And an wlen-sized circle
    xyt = all_thetas(wkernel, theta) #Cylinder of all theta values per point
    ##print xyt.shape, 'xyt shape'

    #unsharp mask the whole data set
    udata = umask(data, kernel)
    
    #Hough transform of same-sized circular window of 1's
    h1 = fast_hough(wkernel, xyt, ntheta)

    #start = time.clock()
    Hthets = []
    Hi = []
    Hj = []
    
    #start0=time.clock()
    dcube = np.repeat(udata[:,:,np.newaxis], repeats=ntheta, axis=2)
    #end0 = time.clock()
    #print 'cube data', end0-start0 
    

    htapp = Hthets.append
    hiapp = Hi.append
    hjapp = Hj.append
    npsum = np.sum

    #Loop: (j,i) are centerpoints of data window.
    datax, datay = data.shape
    for j in xrange(datay):        

        update_progress(j/(datay-1.0)) #For monitoring progress TODO
        if j >= ucntr and j < (datay - ucntr):
            for i in xrange(datax):
                
                if i >= ucntr and i < (datax - ucntr):

                    #TODO
                    if mask is None or (mask is not None and mask[i,j] == 1): #Only necessary for GASS data
                            
                        wcube = dcube[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1,:]   
                        
                        h = npsum(npsum(wcube*xyt,axis=0), axis=0)
                        
                        hout = h/h1 - frac
                        hout[hout<0.0] = 0.0
                    
                        if npsum(hout) > 0:
                            htapp(hout)
                            hiapp(i)
                            hjapp(j)    
    print ''
    return np.array(Hthets), np.array(Hi), np.array(Hj)

#******************************************************************************************
#Lowell's Additions to the Code
#******************************************************************************************

#Create progress meter
def update_progress(progress, message='Progress:'):
    #sys.stdout.write('\r3/3.. [{0}{1}]%'.format('#'*p, ' '*(TEXTWIDTH-p)))
    if progress > 0.0 and progress <= 1.0:
        p = int(TEXTWIDTH*progress/1.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(TEXTWIDTH-p), message))
        sys.stdout.flush()
    elif progress > 0.0 and progress <= 100.0:
        p = int(TEXTWIDTH*progress/100.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(TEXTWIDTH-p), message)) 
        sys.stdout.flush()
    elif progress == 0.0:
        sys.stdout.write('\r{1} [{0}]%'.format(' '*TEXTWIDTH, message))
        sys.stdout.flush()
    else:
        pass ##TODO Progress Bar Failure}}}

def center(filepath, shape=(500, 500)):
    #Returns a cutout from the center of the image
    xy_array, datax, datay = getData(filepath)
    x, y = shape
    if 0 < x < datax and 0 < y < datay:
        import numpy as np
        left = int(datax//2-x//2)
        right = int(datax//2+x//2)
        up = int(datay//2+y//2)
        down = int(datay//2-y//2)
        cutout = np.array(xy_array[left:right, down:up])
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        center_filename = filename+'_center.npy'
        np.save(center_filename, cutout)
        return center_filename
    else:
        return None 


def rht(filepath, output='.'):
    
    #print '1/3.. Loading Data'
    xy_array, datax, datay = getData(filepath)
    #print '1/3.. Successfully Loaded Data!'

    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    print '1/3:: Analyzing', filename, str(datax)+'x'+str(datay)

    #print '2/3.. Setting Params'
    #TODO wrap parameter input
    if filepath.endswith('.fits'): 
        hdu = fits.open(filepath)[0]
        headers = hdu.header['CTYPE1'] + hdu.header['CTYPE2']
        isZEA = any(['ZEA' in x.upper() for x in headers])
    wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask = setParams(xy_array, 51, 10, 0.70, isZEA)
    #print '2/3.. Successfully Set Params!'
    print '2/3:: Line Size:', str(wlen)+',', 'Smoothing Radius:', str(smr)+',', 'Threshold:', str(frac)

    #print '3/3.. Runnigh Hough Transform'
    
    import os

    hi_filename = os.path.join(output, filename + '_hi.npy')
    hj_filename = os.path.join(output, filename + '_hj.npy')
    hthets_filename = os.path.join(output, filename + '_hthets.npy')
    #print '3/3.. Your Data Will Be Saved As:', hi_filename, hj_filename, hthets_filename 
    
    hthets, hi, hj = window_step(xy_array, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask)
    xyt_filename = os.path.join(output, filename + '_xyt.npz')
    putXYT( xyt_filename, hi, hj, hthets)

    np.save(hi_filename, hi)
    np.save(hj_filename, hj)
    np.save(hthets_filename, hthets)

    print '3/3:: Successfully Saved Data!'

def interpret(filepath):
    '''
    Fail-fast! #Optionally can report_errors!
    Using name_hi.npy, name_hj.npy, name_hthets.npy,
    Prodces:
        Backprojection --> name_backproj.npy
        Backprojection Overlayed on Image --> name_overlay(.fits, .jpg, etc) 
        ThetaSpectrum --> name_spectrum.npy
    '''
    try:
        #Read in rht output files
        filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
        hi_filename = filename + '_hi.npy'
        hj_filename = filename + '_hj.npy'
        hthets_filename = filename + '_hthets.npy'
        hi = np.load(hi_filename)
        hj = np.load(hj_filename)
        hthets = np.load(hthets_filename)

        #Spectrum *Length ntheta array of theta power (above the threshold) for whole image*
        spectrum = [np.sum(theta) for theta in hthets]
        spectrum_filename = filename + '_spectrum.npy'
        np.save(spectrum_filename, np.array(spectrum))

        #Backprojection only *Full Size* #Requires image
        image, imx, imy = getData(filepath)
        backproj = np.zeros_like(image)
        coords = zip(hi, hj)
        #for c in range(len(coords)):
            #backproj[coords[c][0]][coords[c][1]] = np.sum(hthets[c]) 
        for c in coords:
            backproj[c[1]][c[0]] = np.sum(hthets[coords.index(c)]) 
        np.divide(backproj, np.sum(backproj), backproj)
        backproj_filename = filename + '_backproj.npy'
        np.save(backproj_filename, np.array(backproj))

        #Overlay of backproj onto image
        #bg_weight = 0.1 #Dims originals image to 10% of the backproj maximum value
        #overlay = np.add(np.multiply(image, bg_weight), np.multiply(image, backproj))
        outline = []
        overlay = copy.deepcopy(image)
        r = 3 #Must be smaller than imx//2 and imy//2
        weight = 1.0/float(2*r+1)**2 #TODO
        for i in range(imx):
            for j in range(imy):
                if backproj[i][j] == 0.0 and np.any(backproj[i-r:i+r, j-r:j+r]):
                    overlay[i][j] = np.sum(backproj[i-r:i+r, j-r:j+r])*weight
        
        #Overlay output
        if filepath.endswith('.fits'):
            #Fits File: http://astropy.readthedocs.org/en/latest/io/fits/#creating-a-new-image-file
            overlay_filename = filename + '_overlay.fits'
            hdu = fits.PrimaryHDU(overlay)
            hdu.writeto(overlay_filename, clobber=True)
        elif filepath.endswith('.npy'):
            overlay_filename = filename + '_overlay.npy'
            np.save(overlay_filename, overlay)
        else:
            import scipy.misc
            import os
            overlay_filename =  filename + '_overlay' + filepath.lstrip(filename)
            #_______________________ #Must reverse overlay y-coords
            scipy.misc.imsave(overlay_filename, overlay[::-1])
        print 'Success'

    except Exception as e:
        #Reported Failure
        raise e
        print 'Failure'
    

def viewer(filepath):
    #Load Libraries
    from matplotlib.pyplot import plot, show, subplot, imshow, title, ylim #contour
    import numpy as np
    
    #Loads in relevant files
    image, imx, imy = getData(filepath)
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'

    print 'Backprojection'
    #contour(np.load(backproj_filename))
    subplot(121)
    imshow(image, cmap='gray')
    ylim(0, imy)
    title(filepath)
    subplot(122)
    imshow(np.load(backproj_filename), cmap='gray')
    ylim(0, imy)
    title(backproj_filename)
    show()

    print 'Theta Spectrum'
    spectrum = np.load(spectrum_filename)
    plot(np.linspace(0.0, 180.0, num=len(spectrum), endpoint=False), spectrum)
    show()

    '''
    if os
        os.startfile(filepath)
    '''


def main(source, output='.'):
    '''
    source: A filename, or the name of a directory containing files to transform (file or dir)
    output_dir: Directory to place output files (default: '.')
    '''
    print '*'*TEXTWIDTH
    print 'Fast Rolling Hough Transform by Susan Clark'
    print '*'*TEXTWIDTH

    #Interpret Filenames from Input
    source = str(source)
    output = str(output)
    pathlist = []
    import os
    if not os.path.isdir(output):
        print 'Invalid output in main(); must be directory.'
        return

    if os.path.isfile(os.path.abspath(source)):
        #Input = File
        pathlist.append(source)
    elif os.path.isdir(source):
        #Input = Directory
        for obj in os.listdir(source):
            if os.path.isfile(os.path.abspath(obj)):
                pathlist.append(obj)
    else:
        #Input = Neither   
        #rht(raw_input('Please enter the pathname of a file to analyze:'), output)
        print 'Invalid source in main(); must be file or directory.'
        return

    #Run RHT Over All Inputs 
    print pathlist
    total = len(pathlist) #TODO
    if (total > 0):
        print 'RHT Started for', source
        done = 0
        update_progress(float(done)/float(total), message='RHT:')
        for path in pathlist:
            print path
            rht(path, output)
            done += 1
            update_progress(float(done)/float(total), message='RHT:')
        print 'RHT Complete in', output
        
