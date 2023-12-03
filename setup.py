from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.spawn import find_executable
import subprocess
import os
import sys
from pathlib import Path
import numpy as np
import GPUtil

PROJECT = os.path.abspath(os.path.dirname(__file__))

# specify installation options
many_nodes = os.environ.get("CUBNM_MANY_NODES") is not None


# detect if there are GPUs
def has_gpus(method='nvcc'):
    if method == 'gputil':
        return len(GPUtil.getAvailable()) > 0
    elif method == 'nvcc':
        # can be useful for compiling code on a
        # non-GPU system for running it later
        # on GPUs
        if find_executable("nvcc"):
                return True
        else:
            return False


gpu_enabled = has_gpus()

# Write the flags to a temporary _flags.py file
with open(os.path.join(PROJECT, "src", "cuBNM", "_setup_flags.py"), "w") as flag_file:
    flag_file.write(
        "\n".join(
            [f"many_nodes_flag = {many_nodes}", 
             f"gpu_enabled_flag = {gpu_enabled}"]
        )
    )

# C++ code compilation
# libraries = ["m", "gsl", "gslcblas", "gomp"] # uncomment for dynamic linking of gsl
libraries = ["m", "gomp"]

_CC = os.environ.get("CC", "gcc") # this will be used for compiling GSL + restoring os $CC and $CXX at the end
_CXX = os.environ.get("CXX", "g++")
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if gpu_enabled:
    print("GPU enabled")
    libraries += ["bnm", "cudart"]
    bnm_ext = Extension(
        "cuBNM.core",
        [os.path.join("src","cpp", "run_simulations.cpp")],
        language="c++",
        extra_compile_args=[  # note that these are not the flags to compile bnm.cu
            "-O3",
            "-m64",
            "-fopenmp",
            "-D NOISE_SEGMENT",
            "-D GPU_ENABLED",
        ],
        libraries=libraries,
        include_dirs=[
            os.path.join(PROJECT, "include"),
            "/usr/lib/cuda/include",
            "/usr/include/cuda",
            os.path.join(PROJECT, "src", "cpp"),
            np.get_include(),
        ],
        library_dirs=[
            "/usr/lib/cuda", 
            os.path.join(PROJECT, "src", "cuda"),
        ],
    )
else:
    bnm_ext = Extension(
        "cuBNM.core",
        [os.path.join("src","cpp", "run_simulations.cpp")],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-m64",
            "-fopenmp",
            "-D NOISE_SEGMENT",
        ],
        libraries=libraries,
        include_dirs=[
            os.path.join(PROJECT,"include"),
            np.get_include(),
        ],
        library_dirs=[],
    )

# extend build_ext to also build GSL (if needed) and compile GPU code
class build_ext_gsl_cuda(build_ext):
    def build_extensions(self):
        # Build GSL (if needed)
        # search for libgsl.a and libgslcblas.a in some common paths
        lib_dirs = ["/usr/lib", "/lib", "/usr/local/lib"] + \
            os.environ.get("LIBRARY_PATH","").split(":") + \
            os.environ.get("LD_LIBRARY_PATH","").split(":") + \
            [os.path.join(os.path.expanduser('~'), '.cuBNM', 'gsl', 'build', 'lib')]
        found_gsl = False
        for lib_dir in lib_dirs:
            if ((lib_dir!='') & os.path.exists(lib_dir)):
                found_lgsl = False
                found_lgslcblas = False
                for f in os.listdir(lib_dir):
                    if os.path.basename(f) == 'libgsl.a': # change to .so for dynamic linking
                        found_lgsl = True
                    elif os.path.basename(f) == 'libgslcblas.a': # same
                        found_lgslcblas = True
                if found_lgsl & found_lgslcblas:
                    found_gsl = True
                    GSL_LIB_DIR = lib_dir
                    print(f"Found libgsl.a and libgslcblas.a in {GSL_LIB_DIR}")
                    break
        if not found_gsl:
            print("Downloading and building GSL")
            try:
                gsl_dir = os.path.join(os.path.expanduser('~'), '.cuBNM', 'gsl')
                os.makedirs(gsl_dir, exist_ok=True)
            except OSError:
                gsl_dir = os.path.join(os.path.abspath(self.build_lib), 'gsl')
                os.makedirs(gsl_dir, exist_ok=True)
            gsl_tar = os.path.join(gsl_dir, 'gsl-2.7.tar.gz')
            gsl_src = os.path.join(gsl_dir, 'gsl-2.7')
            gsl_build = os.path.join(gsl_dir, 'build')
            os.makedirs(gsl_build, exist_ok=True)
            # use gcc (or other default C compilers) as with g++ set as CC
            # GSL compilation fails. Note that this will not affect the compiler
            # of bnm extension as it has already been set to g++
            os.environ["CC"] = _CC
            gsl_setup_commands = [
                f"wget https://mirror.ibcp.fr/pub/gnu/gsl/gsl-2.7.tar.gz -O {gsl_tar}",
                f"cd {gsl_dir} && tar -xf {gsl_tar} &&"
                f"cd {gsl_src} && ./configure --prefix={gsl_build} --enable-shared &&"
                f"make && make install",
                # f"rm {gsl_tar}",
                # f"rm -r {gsl_src}",
            ]
            for command in gsl_setup_commands:
                result = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                # print(f"Command: {command}")
                # print(f"Return Code: {result.returncode}")
                # print(f"stderr:\n{result.stderr}")
            GSL_LIB_DIR = os.path.join(gsl_build, 'lib')
        # Compile CUDA code into libbnm.a
        if gpu_enabled:
            cuda_dir = os.path.join(PROJECT, 'src', 'cuda')

            if 'CUBNM_MANY_NODES' in os.environ:
                add_flags = "-D NOISE_SEGMENT MANY_NODES"
            else:
                add_flags = "-D NOISE_SEGMENT"

            compile_commands = [
                f"nvcc -c -rdc=true --compiler-options '-fPIC' -o {cuda_dir}/bnm_tmp.o {cuda_dir}/bnm.cu "
                    f"-I {PROJECT}/src/cpp -I {PROJECT}/include {add_flags}",
                f"nvcc -dlink --compiler-options '-fPIC' -o {cuda_dir}/bnm.o {cuda_dir}/bnm_tmp.o "
                    f"-L {GSL_LIB_DIR} -lm -lgsl -lgslcblas -lcudart",
                f"rm -f {cuda_dir}/libbnm.a",  # remove the previously created .a
                f"ar cru {cuda_dir}/libbnm.a {cuda_dir}/bnm.o {cuda_dir}/bnm_tmp.o",
                f"ranlib {cuda_dir}/libbnm.a",
            ]

            for command in compile_commands:
                print(command)
                result = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                print(f"Return Code: {result.returncode}")
                if result.stderr:
                    print(f"Standard Error:\n{result.stderr}")

        # Continue with Python extension build
        # add libgsl.a and libgslcblas.a to the compiler objects for
        # their explicit linking
        self.compiler.objects.append(os.path.join(GSL_LIB_DIR, 'libgsl.a'))
        self.compiler.objects.append(os.path.join(GSL_LIB_DIR, 'libgslcblas.a'))
        # self.compiler.add_library_dir(GSL_LIB_DIR) # uncomment for dynamic linking of gsl
        super().build_extensions()

setup(
    ext_modules=[bnm_ext],
    cmdclass={
        'build_ext': build_ext_gsl_cuda,
    },
)

# restore OS's original $CC and $CXX
os.environ['CC'] = _CC
os.environ['CXX'] = _CXX