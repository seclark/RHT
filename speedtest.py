#!/usr/bin/python

#Lowell Schudel
#Execute 'python speedtest.py'

from __future__ import division
import numpy as np
import rht
import time
import timeit
from matplotlib.pyplot import plot, show

WLEN = range(9,250,12)[::-1]
time_a = []
time_b = []

N = 50
DTYPE = np.int

def func_a(two_d, three_d): 
    #IMPLEMENTATION1: Let python figure out the implementation.
    return np.einsum('ijk,ij', three_d, two_d, dtype=DTYPE)

def func_b(two_d, three_d):
    #assert two_d.dtype == np.bool_ 
    #assert three_d.dtype == np.bool_  
    #IMPLEMENTATION3: Broadcast 2D array against 3D stack and AND them together (VERY FAST)
    return np.sum( np.logical_and( two_d.reshape((two_d.shape[0],two_d.shape[1],1)) , three_d) , axis=(1,0), dtype=DTYPE)
    #return np.sum(np.sum( np.logical_and( two_d.reshape((two_d.shape[0],two_d.shape[1],1)) , three_d) , axis=0, dtype=DTYPE), axis=0, dtype=DTYPE)


for wlen in WLEN:
    if not (0 < wlen < 550 and wlen%2):
        time_a.append(0)
        time_b.append(0)
        continue

    print 'Working.. wlen=', wlen
    ntheta = rht.ntheta_w(wlen)
    fake_xyt = np.less(np.random.rand(wlen, wlen, ntheta), 0.5)
    fake_data = np.less(np.random.rand(wlen, wlen), 0.5)
    SETUP = "from __main__ import func_a, func_b, fake_data, fake_xyt"

    #B for Boolean
    #start = time.time()
    #output_b = func_b(fake_data, fake_xyt)
    #time_b.append(time.time()-start)
    time_b.append(timeit.timeit('func_b(fake_data, fake_xyt)', setup=SETUP, number=N)/N)

    #A for Einstein
    fake_xyt = fake_xyt.astype(DTYPE) #______________
    fake_data = fake_data.astype(DTYPE) #_________________
    #start = time.time()
    #output_a = func_a(fake_data, fake_xyt)
    #time_a.append(time.time()-start)
    time_a.append(timeit.timeit('func_a(fake_data, fake_xyt)', setup=SETUP, number=N)/N)

    #Check for equivalence

plot(WLEN, np.divide(time_a, np.power(WLEN, 3)), 'ro', WLEN, np.divide(time_b, np.power(WLEN, 3)), 'bo')
show()

