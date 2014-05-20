from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension("optimized_rw_core", ["optimized_rw_core.pyx"],
        include_dirs = [numpy.get_include()])
]

setup(
    ext_modules = cythonize(extensions)
)
