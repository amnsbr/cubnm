from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
import os
import numpy as np

"""
Pre-requisites:

GSL2.7
> download and unzip & cd to directory
> ./configure --prefix=<target_dir> --enable-shared
> make && make install
"""

# specify if extension needs to be 
many_nodes = os.environ.get("CUBNM_MANY_NODES") is not None
gpu_enabled = os.environ.get("CUDA_VISIBLE_DEVICES") is not None # this probably only works on Juseless and JSC
# Write the value of many_nodes to a temporary file
with open("cuBNM/_flags.py", "w") as flag_file:
    flag_file.write("\n".join(
        [f"many_nodes_flag = {many_nodes}",
        f"gpu_enabled_flag = {gpu_enabled}"]
    ))

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if gpu_enabled:
    print("GPU enabled")
    if many_nodes:
        print("""
        Installing with support for running simulations with high number of nodes 
        which uses __device__ instead of __shared__ memory for S_E history at t-1
        and therefore is slower, but with high N of nodes is needed because
        __shared__ memory is limited
        """)
        libraries = ["m", "gomp", "bnm_many", "cudart"]
    else:
        libraries = ["m", "gomp", "bnm", "cudart"]
    bnm_ext = Extension(
        'cuBNM.core',
        ['cuBNM/cpp/run_simulations.cpp'],
        language='c++',
        extra_compile_args=[
            "-O3",
            "-m64",
            "-fopenmp",
            "-D GPU_ENABLED"
        ],
        libraries=libraries,
        extra_objects=[
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a",
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a",
        ],
        include_dirs=[
            '/data/project/ei_development/tools/gsl_build_shared/include', 
            '/usr/lib/cuda/include',
            '/usr/include/cuda',
            np.get_include(),
            ],
        library_dirs = [".", '/usr/lib/cuda', 'cuBNM/cuda']
    )
else:
    bnm_ext = Extension(
        'cuBNM.core',
        ['cuBNM/cpp/run_simulations.cpp'],
        language='c++',
        extra_compile_args=[
            "-O3",
            "-m64",
            "-fopenmp",
        ],
        libraries=["m", "gomp"],
        extra_objects=[
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a",
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a",
        ],
        include_dirs=[
            '/data/project/ei_development/tools/gsl_build_shared/include', 
            np.get_include(),
            ],
    )

setup(name = 'cuBNM', version = '1.0',
    packages=find_packages(),
    ext_modules = [bnm_ext])

os.remove("cuBNM/_flags.py")