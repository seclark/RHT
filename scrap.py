
import numpy as np
import copy
import math


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


def r_square(theta_square):
    #theta_square must be in radians between [0, 2*pi)
    #Defines the sides of any square with bottome left corner (x, y)
    #Using polar coordinates centered on (x+1/2, y+1/2)
    #With theta_square increasing counterclockwise from the right horizontal
    
    try:
        twoPi = 2.0*np.pi
        while theta_square >= twoPi:
            theta_square -= twoPi
        while theta_square < 0.0:
            theta_square += twoPi
        
        eighth = twoPi/8.0
        if 7*eighth <= theta_square or theta_square < eighth:
            #Right side of square
            return 1.0/(2.0*np.cos(theta_square))
        elif eighth <= theta_square and theta_square < 3*eighth:
            #Top side of square
            return 1.0/(2.0*np.sin(theta_square))
        elif 3*eighth <= theta_square and theta_square < 5*eighth:
            #Left side of square
            return -1.0/(2.0*np.cos(theta_square))
        else: #elif 5*eighth <= theta_square and theta_square < 7*eighth:
            #Bottom side of square
            return -1.0/(2.0*np.sin(theta_square))
    
    except Exception:
        return None

	
def ntheta_fast(wlen):
	import math
	import numpy
	return math.ceil( (numpy.pi*numpy.sqrt(2)*((wlen-1)/2.0)) )

def ntheta_slow(x, y):
	slope_set = []
	for i in range(1, x):
		for j in range(1, y):
			slope_set.append(j/i)
	#Counts the number of unique slopes in image quadrant I
	#Doubles to account for quadrant II, +1 is for y-axis
	return 1 + 2*len(set(slope_set)) 
