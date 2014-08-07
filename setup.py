

from distutils.core import setup
from Cython.Build import cythonize

setup(name='cython_hough', ext_modules = cythonize("cython_hough.pyx"),)