
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def fast_hough(np.ndarray in_arr, np.ndarray xyt): 
    return np.einsum('ijk,ij', xyt, in_arr)

