#!/usr/bin/python

#Lowell Schudel
#GPU Test

import numpy
from numbapro import vectorize

# Create a ufunc
@vectorize(['float32(float32, float32)',
            'float64(float64, float64)'])
def sum(a, b):
    return a + b

# Use the ufunc
a = numpy.arange(10)
b = numpy.arange(10)
result = sum(a, b)      # call the ufunc

print("a = %s" % a)
print("b = %s" % b)
print("sum = %s" % result)


htapp = Hthets.append
hiapp = Hi.append
hjapp = Hj.append
npsum = np.sum

#Loop: (j,i) are centerpoints of data window.
datax, datay = data.shape

for j in xrange(datay):

    if j >= ucntr and j < (datay - ucntr):
        
        for i in xrange(datax):
        
            if i >= ucntr and i < (datax - ucntr):
    
                wcube = dcube[j-wcntr:j+wcntr+1, i-wcntr:i+wcntr+1,:]

                h = npsum(npsum(wcube*xyt,axis=0), axis=0)
                
                hout = h/h1 - frac
                hout[hout<0.0] = 0.0
            
                if npsum(hout) > 0:
                    htapp(hout)
                    hiapp(i)
                    hjapp(j)