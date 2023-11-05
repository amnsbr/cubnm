from setuptools import setup, Extension
import os
import numpy as np

"""
Pre-requisites:

GSL2.7
> download and unzip & cd to directory
> ./configure --prefix=<target_dir> --enable-shared
> make && make install

libks
> git clone ... && cd
> make
"""

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
gpu_enabled = True

if gpu_enabled:
    bnm_ext = Extension(
        'cuBNM.core',
        ['cuBNM/cpp/run_simulations.cpp'],
        language='c++',
        extra_compile_args=[
            "-O3",
            "-m64",
            "-fopenmp",
        ],
        libraries=["m", "gomp", "bnm", "cudart"],
        extra_objects=[
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a",
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a",
            "/data/project/ei_development/tools/libks/libks.so",
        ],
        include_dirs=[
            '/data/project/ei_development/tools/gsl_build_shared/include', 
            '/data/project/ei_development/tools/libks/include',
            '/usr/lib/cuda/include',
            '/usr/include/cuda',
            np.get_include(),
            ],
        library_dirs = [".", '/usr/lib/cuda', 'cuBNM/cuda']
    )
else:
    raise NotImplementedError("The package currently does not support CPU simulations")

setup(name = 'cuBNM', version = '1.0',  \
   ext_modules = [bnm_ext])