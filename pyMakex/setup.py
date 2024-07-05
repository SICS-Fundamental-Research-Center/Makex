#from distutils.core import setup, Extension
from setuptools import setup, Extension

# define the extension module
pyCFLogic = Extension('pyMakex',
                      language='c++',
                      sources=['pyMakex.cpp'],
                      extra_compile_args=['-lm -O3  -ffast-math']+['-fopenmp'],
                      extra_link_args=['-lgomp'],
                      include_dirs=['./include/'])


# Run the setup
setup(name='pyMakex',
      version='1.0',
      ext_modules=[pyCFLogic])
