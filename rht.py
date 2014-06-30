#!/usr/bin/python

# FAST ROLLING HOUGH TRANSFORM
# ulen : length of unsharp mask square. Must be at least wlen + smr/2

#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------
from __future__ import division
import numpy as np
import scipy
import scipy.ndimage
import math
from astropy import wcs
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import copy
import sys
import string


#-----------------------------------------------------------------------------------------
#Parameters
#-----------------------------------------------------------------------------------------
TEXTWIDTH = 60 #Width of some displayed text objects
WLEN = 50 #Diameter of a 'window' to be evaluated at one time
FRAC = 0.75 #fraction (percent) of one angle that must be 'lit up' to be counted
SMR = 5 #smoothing radius of unsharp mask function

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
        #result = '\n'.join(['*'*TEXTWIDTH, string.center(strings, TEXTWIDTH, ' '), '*'*TEXTWIDTH])
    else:
        result = announcement(str(strings))
    return result

def announce(strings):
    print announcement(strings)

def update_progress(progress, message='Progress:'):
    #Create progress meter
    if progress > 0.0 and progress <= 1.0:
        p = int(TEXTWIDTH*progress/1.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(TEXTWIDTH-p), message))
        sys.stdout.flush()
        if p == TEXTWIDTH:
            print ''
    elif progress > 0.0 and progress <= 100.0:
        p = int(TEXTWIDTH*progress/100.0) 
        sys.stdout.write('\r{2} [{0}{1}]%'.format('#'*p, ' '*(TEXTWIDTH-p), message)) 
        sys.stdout.flush()
        if p == TEXTWIDTH:
            print ''
    elif progress == 0.0:
        sys.stdout.write('\r{1} [{0}]%'.format(' '*TEXTWIDTH, message))
        sys.stdout.flush()
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
        gassslice = scipy.ndimage.imread(filepath, True)[::-1] #Makes B/W array, reversing y-coords

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



def makemask(wkernel, gassslice):
    # The following is specific to a certain data set (the Parkes Galactic All-Sky Survey)
    # which was in a Zenith-Equal-Area projection. This projects the sky onto a circle, and so 
    # makemask just makes sure that nothing outside that circle is counted as data.

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
    
    gmconv = scipy.ndimage.filters.correlate(gm, weights=wkernel)
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
    outdata = scipy.ndimage.filters.correlate(data, weights=inkernel)
    
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
                    if mask is None or (mask is not None and mask[i,j] == 1): 
                            
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

#-----------------------------------------------------------------------------------------
#Interactive Functions
#-----------------------------------------------------------------------------------------


def center(filepath, shape=(500, 500)):
    #Returns a cutout from the center of the image
    xy_array, datax, datay = getData(filepath)
    x, y = shape
    if 0 < x < datax and 0 < y < datay:
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
    
    excluded_file_endings = ['_xyt.npz', '_backproj.npy', '_spectrum.npy']
    if any([filepath.endswith(e) for e in excluded_file_endings]):
        return False

    #print '1/3.. Loading Data'
    xy_array, datax, datay = getData(filepath)
    #print '1/3.. Successfully Loaded Data!'

    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    print '1/3:: Analyzing', filename, str(datax)+'x'+str(datay)

    #print '2/3.. Setting Params'
    #TODO wrap parameter input
    isZEA = False
    if filepath.endswith('.fits'): 
        hdu = fits.open(filepath)[0]
        headers = hdu.header['CTYPE1'] + hdu.header['CTYPE2']
        isZEA = any(['ZEA' in x.upper() for x in headers])
    wlen, frac, smr, ucntr, wcntr, ntheta, dtheta, theta, mask = setParams(xy_array, 51, 10, 0.70, isZEA)
    #print '2/3.. Successfully Set Params!'
    print '2/3:: Window Diameter:', str(wlen)+',', 'Smoothing Radius:', str(smr)+',', 'Threshold:', str(frac)

    #print '3/3.. Runnigh Hough Transform'
    hthets, hi, hj = window_step(xy_array, wlen, frac, smr, ucntr, wcntr, theta, ntheta, mask)
    
    xyt_filename = os.path.join(output, filename + '_xyt.npz')
    putXYT(xyt_filename, hi, hj, hthets)

    #hi_filename = os.path.join(output, filename + '_hi.npy')
    #hj_filename = os.path.join(output, filename + '_hj.npy')
    #hthets_filename = os.path.join(output, filename + '_hthets.npy')
    #np.save(hi_filename, hi)
    #np.save(hj_filename, hj)
    #np.save(hthets_filename, hthets)

    print '3/3:: Successfully Saved Data As', xyt_filename
    return True

def interpret(filepath, force=False):
    '''
    Using name_xyt.npz INSTEAD OF name_hi.npy, name_hj.npy, name_hthets.npy,
    Prodces:
        Backprojection --> name_backproj.npy
        Backprojection Overlayed on Image --> name_overlay(.fits, .jpg, etc) 
        ThetaSpectrum --> name_spectrum.npy
    '''
    #Read in rht output files
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    xyt_filename = filename + '_xyt.npz'

    #Makes sure relevant files are present! #TODO See if this works!
    required_files = [xyt_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing:
        if force:
            #Runs rht(filepath), since that has not been done
            rht(filepath)
        else:
            #Warns user against interpreting file
            print 'Warning: required files not present for interpret(filepath)...'

    #Proceed with iterpreting
    hi, hj, hthets = getXYT(xyt_filename, rebuild=False)
    '''
    hi_filename = filename + '_hi.npy'
    hj_filename = filename + '_hj.npy'
    hthets_filename = filename + '_hthets.npy'
    hi = np.load(hi_filename)
    hj = np.load(hj_filename)
    hthets = np.load(hthets_filename)
    '''

    #Spectrum *Length ntheta array of theta power (above the threshold) for whole image*
    spectrum = [np.sum(theta) for theta in hthets]
    spectrum_filename = filename + '_spectrum.npy'
    np.save(spectrum_filename, np.array(spectrum))

    #Backprojection only *Full Size* #Requires image
    image, imx, imy = getData(filepath)
    backproj = np.zeros_like(image)
    coords = zip(hi, hj)
    for c in range(len(coords)):
        backproj[coords[c][0]][coords[c][1]] = np.sum(hthets[c]) 
    #for c in coords: #SLOW VERSION, EQUIVALENT TO ABOVE
        #backproj[c[1]][c[0]] = np.sum(hthets[coords.index(c)]) 
    np.divide(backproj, np.sum(backproj), backproj)
    backproj_filename = filename + '_backproj.npy'
    np.save(backproj_filename, np.array(backproj))

    '''
    #Overlay of backproj onto image
    #bg_weight = 0.1 #Dims originals image to 1/10 of the backproj maximum value
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
        overlay_filename =  filename + '_overlay' + filepath.lstrip(filename)
        #_______________________ #Must reverse overlay y-coords
        scipy.misc.imsave(overlay_filename, overlay[::-1])
    '''

    print 'Success'

    
def viewer(filepath, force=False):
    
    #Loads in relevant files
    image, imx, imy = getData(filepath)
    filename = '.'.join( filepath.split('.')[ 0:filepath.count('.') ] )
    backproj_filename = filename + '_backproj.npy'
    spectrum_filename = filename + '_spectrum.npy'

    #Makes sure relevant files are present! #TODO See if this works!
    required_files = [backproj_filename, spectrum_filename]
    any_missing = any([not os.path.isfile(f) for f in required_files])
    if any_missing:
        if force:
            #Interprets file, since that has not been done
            interpret(filepath, force=True)
        else:
            #Warns user against interpreting file
            print 'Warning: required files not present for viewer(filepath)...'
    
    print 'Backprojection'
    #contour(np.load(backproj_filename))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.ylim(0, imy)
    plt.title(filepath)
    plt.subplot(122)
    plt.imshow(np.load(backproj_filename), cmap='gray')
    plt.ylim(0, imy)
    plt.title(backproj_filename)
    plt.show()
    #___________________________________TODO SAVE PLOT
    plt.clf()
    plt.cla()
    plt.close()

    print 'Theta Spectrum'
    spectrum = np.load(spectrum_filename)
    ntheta = len(spectrum)
    plt.plot(np.linspace(0.0, 180.0, num=ntheta, endpoint=False), spectrum)
    #___________________________________TODO SAVE PLOT
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #Polar plot of theta power
    print 'Linearity'
    r = np.append(spectrum, spectrum) 
    t = np.linspace(0.0, 2*np.pi, num=len(r), endpoint=False)
    plt.polar(t, r)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #Clear all
    plt.clf()
    plt.cla()
    plt.close('all')


def main(source=None, display=False):
    '''
    source: A filename, or the name of a directory containing files to transform
    display: Boolean flag determining if the input is to be interpreted and displayed
    '''
    #Ensure that the input is a non-None string
    while source is None or source != str(source): #TODO Fix escape char bug
        try:
            source = raw_input('Please enter the name of a file or directory to transform:')
        except:
            source = None
    
    #Interpret Filenames from Input
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
        #_____________________________TODO
        print 'Invalid source encountered in main(); must be file or directory.'
        return

    #Run RHT Over All Inputs 
    announce('Fast Rolling Hough Transform by Susan Clark')
    print 'RHT Started for:', source
    total = len(pathlist)
    if total == 0:
        print 'Error'#_____________________________TODO
    elif total == 1:
        if (display):
            viewer(pathlist[0], force=True)
        else:
            rht(pathlist[0])
        print 'RHT Complete!'
    else:
        for path in pathlist:
            if (display):
                viewer(path, force=True)
            else:
                rht(path)
        print 'RHT Complete!'
    return
        
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
  
 -h, help ==> Displays this message
 >>>python rht.py help

 -p, params ==> Displays Default Params
 >>>python rht.py -p
 
MULTIPLE ARGS:
 Creates 'dirname/filename_xyt.npz' for each input image
 1st ==> Path to input file or directory
 2nd:nth ==> Named inputs controlling params and flags
  Flags: 
  -d  #Sets whether ouput is displayed
  Params:
  -wlen=value  #Sets window diameter
  -smr=value  #Sets smoothing radius
  -frac=value  #Sets theta power threshold'''
    
    argn = len(sys.argv)
    if argn == 1:
        #Displays the README file   
        README = 'README'
        readme = open(README, 'r')
        print readme.read(2000) 
        if len(readme.read(1)) == 1:
            print ''
            print '...see', README, 'for more information...'
            print ''
        readme.close()

    elif argn == 2:
        #Parses input for single argument flags
        source = sys.argv[1]
        if source.lower() in ['help', '-help', 'h', '-h']:
            announce(help)
        elif source.lower() in ['params', 'param', 'p', '-p', '-params', '-param']:
            params = ['Default RHT Parameters:']
            params.append('wlen = '+str(WLEN))
            params.append('smr = '+str(SMR))
            params.append('frac = '+str(FRAC))
            announce(params) #__________________TODO
        else:
            main(source)

    else:
        source = sys.argv[1]
        args = sys.argv[2:]
        for arg in args:
            DISPLAY = False
            if '=' not in arg:
                #FLAGS which DO NOT carry values 
                if arg.lower() in ['d', '-d', 'display', '-display' ]:
                    DISPLAY = True
                    #TODO_________________________________Turn on Display flag!
                else:
                    print 'UNKNOWN FLAG:', arg

            else:
                #PARAMETERS which DO carry values
                argname = arg.lower().split('=')[0]
                argval = arg.lower().split('=')[1] #TODO Handle errors
                if argname in ['w', 'wlen', '-w', '-wlen']:
                    WLEN = argval
                elif argname in ['s', 'smr', '-s', '-smr']:
                    SMR = argval
                elif argname in ['f', 'frac', '-f', '-frac']:
                    FRAC = argval
                else:
                    print 'UNKNOWN PARAMETER:', arg

        main(source, display=DISPLAY)

    exit()
        
        

#-----------------------------------------------------------------------------------------
#Attribution
#-----------------------------------------------------------------------------------------

#This is the Rolling Hough Transform, described in Clark, Peek, Putman 2014 (arXiv:1312.1338).
#Modifications to the RHT have been made by Lowell Schudel, CC'16.
