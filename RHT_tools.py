from astropy.io import fits
import numpy as np
import math

def get_thets(wlen, save = True):
    """
    Determine the values of theta for RHT output. These are determined by the window length (wlen)
    by Equation 2 in Clark+ 2014.
    """

    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))
    print 'ntheta is ', ntheta
    dtheta = np.pi/ntheta
    
    #Thetas for binning   
    thetbins = dtheta*np.arange(0, ntheta+2)
    thetbins = thetbins - dtheta/2
    
    # thets for plotting
    thets = np.arange(0, np.pi, dtheta)
    
    if save == True:
        np.save('thets_w'+str(wlen)+'.npy', thets)
    
    return thets

def get_RHT_data(wlen = 75, smr = 15, thresh = 70, xyt_filename = "filename.fits"):
        
    hdu_list = fits.open(xyt_filename, mode='readonly', memmap=True, save_backup=False, checksum=True) #Allows for reading in very large files!
    print "loading data from ", xyt_filename
    header = hdu_list[0].header
    data = hdu_list[1].data
    ipoints = data['hi'] 
    jpoints = data['hj'] 
    hthets = data['hthets']
    
    naxis1 = header["NAXIS1"]
    naxis2 = header["NAXIS2"]
    
    return ipoints, jpoints, hthets, naxis1, naxis2

def grid_QU_RHT(wlen = 75, smr = 15, thresh = 70, xyt_filename = "filename.fits", root = "/your_path/", fn = "filename", save = True):
    
    # Load RHT data
    ipoint, jpoints, hthets, naxis1, naxis2 = get_RHT_data(wlen = wlen, smr = smr, thresh = thresh, xyt_filename = xyt_filename)
  
    # Values of theta for RHT output
    thets = get_thets(wlen, save = False)
    
    URHT = np.zeros((naxis2, naxis1), np.float_)
    QRHT = np.zeros((naxis2, naxis1), np.float_)
    URHTsq = np.zeros((naxis2, naxis1), np.float_)
    QRHTsq = np.zeros((naxis2, naxis1), np.float_)
    intrht = np.zeros((naxis2, naxis1), np.float_)
    
    print naxis2, naxis1
    
    jipoints = zip(jpoints, ipoints)
    jih = dict(zip(jipoints, hthets))
    
    # Calculate Q and U from the RHT output for each point in the image
    for s in jih.keys():
        QRHT[s], URHT[s], QRHTsq[s], URHTsq[s] = get_QU_RHT_unnorm(jih[s], thets, sqerror = True)
        intrht[s] = np.sum(jih[s])
        
    if save == True:       
        np.save("U_RHT_"+fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", URHT)
        np.save("Q_RHT_"+fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", QRHT)
        np.save("U_RHT_sq_"+fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", URHTsq)
        np.save("Q_RHT_sq_"+fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", QRHTsq)
        np.save("intrht_"+fn+"_w"+str(wlen)+"_s"+str(smr)+"_t"+str(thresh)+".npy", intrht) 

