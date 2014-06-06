
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

def interpret(filepath=None, interactive=False):
    '''
    Using name_hi.npy, name_hj.npy, name_hthets.npy,
    Prodces:
        Backprojection --> name_backproj.npy
        Backprojection Overlayed on Image --> name_overlay.npy 
        ThetaSpectrum --> name_spectrum.npy
    '''

    if not interactive:
        try:
            filename = filepath.split(".")[0]
            hi_filename = filename + '_hi.npy'
            hj_filename = filename + '_hj.npy'
            hthets_filename = filename + '_hthets.npy'
            import numpy as np
            hi = np.load(hi_filename)
            hj = np.load(hj_filename)
            hthets = np.load(hthets_filename)
            print 'Loaded'
            
            #Backprojection *Full Size* #Requires image
            image, imx, imy = getData(filepath)
            full_tflat_xy = np.zeros_like(image)
            coords = zip(hi, hj)
            #for c in range(len(coords)):
                #full_tflat_xy[coords[c][0]][coords[c][1]] = np.sum(hthets[c]) 
            for c in coords:
                full_tflat_xy[c[0]][c[1]] = np.sum(hthets[coords.index(c)]) 
            from matplotlib.pyplot import plot, show, contour
            print 'Boolean Backprojection'
            contour(full_tflat_xy)
            show()
            exit()

            backproj_filename = filename + '_backproj.npy'
            np.save(backproj_filename, np.array(small_tflat_xy))
            print 'Backproj'

            '''
            #Backprojection *Minimum XY Size, coords offset by low*
            low = [min(hi), min(hj)]
            high = [max(hi), max(hj)]
            small_tflat_xy = np.zeros(np.add(np.subtract(high, low), (1, 1)))
            #print small_tflat_xy.shape, 'small'
            coords = np.subtract(zip(hi, hj), low)
            #print coords.shape, 'coords'
            for c in range(len(coords)):
                #print coords[c][0], coords[c][1]
                small_tflat_xy[coords[c][0]][coords[c][1]] = np.sum(hthets[c])
            backproj_filename = filename + '_backproj.npy'
            np.save(backproj_filename, np.array(small_tflat_xy))
            print 'Backproj'
            
            #Overlay *Image coords*
            image, imx, imy = getData(filepath)
            #np.divide(image, np.amax(image)) #TODO: Image Weighting to 1?
            large_tflat_xy = np.zeros_like(image)
            small_shape = small_tflat_xy.shape
            for a in range(small_shape[0]):
                for b in range(small_shape[1]):
                    large_tflat_xy[a+low[0]][b+low[1]] = small_tflat_xy[a][b]
            weight = 1.0 #TODO: Weight by powers of large_tflat_xy
            overlay = np.multiply(image, np.multiply(large_tflat_xy, weight))
            overlay_filename = filename + '_overlay.npy'
            np.save(overlay_filename, np.array(overlay))
            '''

            #Spectrum *Length ntheta array of theta power for whole image*
            spectrum = [sum(theta) for theta in hthets]
            spectrum_filename = filename + '_spectrum.npy'
            np.save(spectrum_filename, np.array(spectrum))


        except Exception as e:
            print e.args #TODO
            pass #Silent, fast failure
    else:   
        print '*'*TEXTWIDTH
        print 'Rolling Hough Transform Interpreter by Lowell Schudel'
        print '*'*TEXTWIDTH
        
        '''
        Input Handling

        Failures:
        0- Bad filepath
            0.1-
        1- Output files not found
            1.1- Did not choose to reananlyze
            1.2- Generated output files, reinterpreted
            1.3- Failed to find image file
        2- Data reading failure
            2.1

        Exits:
        0- No image filepath entered
        1- Outputs not found, no new analysis
        '''
        from os.path import isfile
        #Filename Assignment
        if filepath==None:
            try:
                filepath = raw_input('Please enter the relative path of a file to analyze:')         
            except EOFError:
                exit('Exiting interpret: 0') #Exit 0
            
        try:
            filename = filepath.split(".")[0]
        except IndexError:
            print 'Filename does not appear to have an extension'
        hi_filename = filename + '_hi.npy'
        hj_filename = filename + '_hj.npy'
        hthets_filename = filename + '_hthets.npy'
        
        
        
        if not(isfile(hi_filename) and isfile(hj_filename) and isfile(hthets_filename)): 
            print 'Output files for this image were not found.'
            from distutils.util import strtobool
            try:
                redo = strtobool(raw_input('Would you like to reanalyze the image? y/[n]'))
            except ValueError:
                #No choice
                redo = False #Failure 1.1 
            except EOFError:
                #Default choice
                redo = False 

            if redo: 
                if isfile(filepath):
                    main(filepath, silent=True)
                    print 'File analyzed successfully. Reinterpreting...' #Failure 1.2
                    interpret(filepath)
                else:
                    print 'Nonexistant image file, please try another.' #Failure 1.3
            else:
                exit('Exiting interpret: 1') #Exit 1

        else: 
            from numpy import load
            try:
                hi = np.load(hi_filename)
                hj = np.load(hj_filename)
                hthets = np.load(hthets_filename)
            except IOError: 
                print 'One or more output files are invalid' #Failure 2.1



