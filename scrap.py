
import numpy as np
import copy

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

def circle(w, r):
	#
	square = np.ones((w, w), np.int_) #Square of 1s
	kernel = circ_kern(square, r)
	return kernel #Produces one circle

def ring(w, r1, r2):
	#
	square = np.ones((w, w), np.int_) #Square of 1s
	kernel1 = circ_kern(square, r1)
	kernel2 = circ_kern(square, r2)
	return kernel1^kernel2 #XORs two circ_kerns together
	
