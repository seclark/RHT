from __future__ import division, print_function
from astropy.io import fits
import numpy as np
import math

def get_thets(wlen, save = True, returnbins = False, verbose = False):
    """
    Determine the values of theta for RHT output. These are determined by the window length (wlen)
    by Equation 2 in Clark+ 2014.
    """

    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))
    if verbose:
        print('ntheta is {}'.format(ntheta))
    dtheta = np.pi/ntheta
    
    #Thetas for binning   
    thetbins = dtheta*np.arange(0, ntheta+2)
    thetbins = thetbins - dtheta/2
    
    # thets for plotting
    thets = np.linspace(0.0, np.pi, ntheta, endpoint=False)
    
    if save == True:
        np.save('thets_w'+str(wlen)+'.npy', thets)
    
    if returnbins:
        return thets, thetbins
    else:
        return thets

def get_RHT_data(xyt_filename = "filename.fits"):
    """
    Loads in RHT data from file.
    """
        
    hdu_list = fits.open(xyt_filename, mode='readonly', memmap=True, save_backup=False, checksum=True) #Allows for reading in very large files!
    print("loading data from ", xyt_filename)
    header = hdu_list[0].header
    data = hdu_list[1].data
    ipoints = data['hi'] 
    jpoints = data['hj'] 
    hthets = data['hthets']
    
    naxis1 = header["NAXIS1"]
    naxis2 = header["NAXIS2"]
    wlen   = header['WLEN']
    smr    = header['SMR']
    thresh = header['FRAC']
    
    return ipoints, jpoints, hthets, naxis1, naxis2, wlen, smr, thresh

def grid_QU_RHT(xyt_filename = "filename.fits", output_fn = "output_filename", save = True):
    
    # Load RHT data
    ipoints, jpoints, hthets, naxis1, naxis2, wlen, smr, thresh = get_RHT_data(xyt_filename = xyt_filename)
  
    # Values of theta for RHT output
    thets = get_thets(wlen, save = False)
    
    URHT = np.zeros((naxis2, naxis1), np.float_)
    QRHT = np.zeros((naxis2, naxis1), np.float_)
    URHTsq = np.zeros((naxis2, naxis1), np.float_)
    QRHTsq = np.zeros((naxis2, naxis1), np.float_)
    intrht = np.zeros((naxis2, naxis1), np.float_)
    
    # Create dictionary of (i, j) points and their corresponding hthets array
    jipoints = zip(jpoints, ipoints)
    jih = dict(zip(jipoints, hthets))
    
    # Calculate Q and U from the RHT output for each point in the image
    for s in jih.keys():
        QRHT[s], URHT[s], QRHTsq[s], URHTsq[s] = get_QU_RHT_unnorm(jih[s], thets, sqerror = True)
        intrht[s] = np.sum(jih[s])
    
    # Save data using defined output_filename
    if save == True:       
        np.save("U_RHT_"+output_fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", URHT)
        np.save("Q_RHT_"+output_fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", QRHT)
        np.save("U_RHT_sq_"+output_fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", URHTsq)
        np.save("Q_RHT_sq_"+output_fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", QRHTsq)
        np.save("intrht_"+output_fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", intrht) 
        
    return QRHT, URHT, URHTsq, QRHTsq, intrht
        
def get_QU_RHT_unnorm(hthets, thets, sqerror = True):
    """ 
    Return QHRT, URHT from single hthets
    """

    QRHT = np.sum(np.cos(2*thets)*hthets)
    URHT = np.sum(np.sin(2*thets)*hthets)
    
    # These values are useful for calculating RHT error
    if sqerror == True:
        QRHTsq = np.sum((np.cos(2*thets))**2*hthets)
        URHTsq = np.sum((np.sin(2*thets))**2*hthets)
    
    # NaN anything which is bad data
    if np.sum(hthets) <= 0:
        QRHT = None
        URHT = None
        QRHTsq = None
        URHTsq = None
    
    if sqerror == True:
        return QRHT, URHT, QRHTsq, URHTsq
    
    else:
        return QRHT, URHT

