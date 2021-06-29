import numpy as np
from rht import rht
import h5py
import time
from astropy.io import fits
import scipy.ndimage as ndimage

def unsharp_mask(data, smr=2, rms_cutoff=None):
    umask_data_bool = rht.umask(data, smr)
    if type(rms_cutoff) in (float, int):
        ## If rms_cutoff is float or integer
        ## Set it as constant cutoff level throughout the entire map
        mask_im = (data >= rms_cutoff)
        umask_data_bool = np.logical_and(umask_data_bool, mask_im)
    elif type(rms_cutoff) in (str, np.ndarray):
        if type(rms_cutoff) == str:
            ## If rms_cutoff is a string (presumably path to .FITS file)
            ## Read in the .FITS file using astropy, use as map of cutoff level
            rms_map = fits.getdata(rms_cutoff)
        else:
            ## If rms_cutoff is a numpy.ndarray
            ## Use as map of cutoff level directly
            rms_map = rms_cutoff
        if rms_map.shape != data.shape:
            ## Shape of cutoff level map is not the same as data map
            raise RuntimeError('shape of rms_cutoff map does not match that of data map')
        mask_im = (data >= rms_map)
        umask_data_bool = np.logical_and(umask_data_bool, mask_im)
    elif rms_cutoff == None:
        pass
    else:
        ## Unsupported data type --> TypeError
        raise TypeError('rms_cutoff must be \'None\', float, int, str, or numpy.ndarray')
    umask_data = np.zeros(umask_data_bool.shape)
    umask_data[umask_data_bool] = 1. # instead of bitmask
    
    return umask_data
    
def convRHT(datafn, wlen=11, smr=2, thresh=0.7, outroot="", outname="name", verbose=False, rms_cutoff=None):
    """
    Convolution-based implementation of the RHT
    Utilizes the 'xyt' array from the original code: 
    xyt defines the translation from map-space to theta-space for a given window size

    This implementation convolves the map with each theta bin of xyt.
    Output identical to original implementation, but much faster.
    Output now written to an hdf5 file rather than FITS.
    """
    
    ## Check if datafn is a string or numpy.ndarray
    ## For string, read in .FITS files using astropy
    ## For numpy.ndarray, use it as data directly
    if type(datafn) == str:
        ## If datafn is a string (presumably path to .FITS file)
        ## Read in the .FITS file using astropy
        data = fits.getdata(datafn)
    elif type(datafn) == np.ndarray:
        ## If datafn is a numpy.ndarray
        ## Use as data directly
        data = datafn
    else:
        ## Unsupported data type --> TypeError
        raise TypeError('datafn must be str or numpy.ndarray')
    umask_data = unsharp_mask(data, smr=smr, rms_cutoff=rms_cutoff)
    
    # Geometry
    datay, datax = data.shape
    ntheta = np.int(np.ceil( np.pi*(wlen-1)/np.sqrt(2.0) ))  
    theta, dtheta = np.linspace(0.0, np.pi, ntheta, endpoint=False, retstep=True)        
    
    # Cylinder of all lit pixels along a theta value
    xyt = rht.all_thetas(wlen=wlen, theta=theta, original=True) 
    
    # Store output in hdf5 cube
    out_fn = outroot+"{}_wlen{}_smr{}_thresh{}.h5".format(outname, wlen, smr, thresh)
    with h5py.File(out_fn, 'w') as f:
        f.create_dataset(name='rht_cube', data=np.zeros((datay, datax, ntheta), np.float_), compression="gzip")
        f['rht_cube'].attrs['wlen'] = wlen
        f['rht_cube'].attrs['smr'] = smr
        f['rht_cube'].attrs['thresh'] = thresh

    for _i in np.arange(ntheta):
        cdata = ndimage.convolve(umask_data, xyt[:, :, _i]/np.nansum(xyt[:, :, _i])) - thresh
        cdata[np.where(cdata < 0)] = 0.
        if verbose:
            print("{} of {} finished".format(_i+1, ntheta))
        
        with h5py.File(out_fn, 'r+') as f:
            f['rht_cube'][:, :, _i] = cdata

    

if __name__ == "__main__":
    
    # demo on test data provided in RHT repo
    dataroot = "testdata/"
    outname = "testim_tesla_small"
    data_fn = dataroot + outname + ".fits"
    data = fits.getdata(data_fn)
    
    wlen = 21
    smr = 2
    thresh = 0.7
    verbose = True
    
    time0 = time.time()
    convRHT(data_fn, wlen=wlen, smr=smr, thresh=thresh, outroot=dataroot, outname=outname, verbose=verbose)
    time1 = time.time()
    
    print("Total time {} minutes".format((time1 - time0)/60.))
    
