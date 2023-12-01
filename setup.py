from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.spawn import find_executable
import subprocess
import os
import numpy as np
import GPUtil

PROJECT = os.path.abspath(os.path.dirname(__file__))

# specify installation options
many_nodes = os.environ.get("CUBNM_MANY_NODES") is not None


# detect if there are GPUs
def has_gpus(method='gputil'):
    if method == 'gputil':
        return len(GPUtil.getAvailable()) > 0
    else:
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

# extend build_ext to also compile GPU code
class build_ext_w_cuda(build_ext):
    def build_extensions(self):
        ##  TODO: Build GSL
        # subprocess.run(["./configure", "--prefix=" + gsl_prefix, "--enable-shared"], cwd="path/to/gsl")
        # subprocess.run(["make", "install"], cwd="path/to/gsl")

        # Compile CUDA code into libbnm.a
        if gpu_enabled:
            cuda_dir = os.path.join(PROJECT, 'src', 'cuda')

            if 'CUBNM_MANY_NODES' in os.environ:
                add_flags = "-D NOISE_SEGMENT MANY_NODES"
            else:
                add_flags = "-D NOISE_SEGMENT"

            compile_commands = [
                f"nvcc -c -rdc=true --compiler-options '-fPIC' -o {cuda_dir}/bnm_tmp.o {cuda_dir}/bnm.cu "
                f"-I {PROJECT}/src/cpp {add_flags} -I /data/project/ei_development/tools/gsl_build_shared/include",
                f"nvcc -dlink --compiler-options '-fPIC' -o {cuda_dir}/bnm.o {cuda_dir}/bnm_tmp.o "
                f"/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a "
                f"/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a -lm -lcudart",
                f"rm -f {cuda_dir}/libbnm.a",
                f"ar cru {cuda_dir}/libbnm.a {cuda_dir}/bnm.o {cuda_dir}/bnm_tmp.o",
                f"ranlib {cuda_dir}/libbnm.a"
            ]

            for command in compile_commands:
                subprocess.run(command, shell=True)

        # Continue with Python extension build
        super().build_extensions()

# C++ code compilation
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if gpu_enabled:
    print("GPU enabled")
    libraries = ["m", "gomp", "bnm", "cudart"]
    bnm_ext = Extension(
        "cuBNM.core",
        ["src/cpp/run_simulations.cpp"],
        language="c++",
        extra_compile_args=[  # note that these are not the flags to compile bnm.cu
            "-O3",
            "-m64",
            "-fopenmp",
            "-D NOISE_SEGMENT",
            "-D GPU_ENABLED",
        ],
        libraries=libraries,
        extra_objects=[
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a",
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a",
        ],
        include_dirs=[
            "/data/project/ei_development/tools/gsl_build_shared/include",
            "/usr/lib/cuda/include",
            "/usr/include/cuda",
            os.path.join(PROJECT,"src", "cpp"),
            np.get_include(),
        ],
        library_dirs=[".", "/usr/lib/cuda", os.path.join(PROJECT,"src", "cuda")],
    )
else:
    bnm_ext = Extension(
        "cuBNM.core",
        ["src/cpp/run_simulations.cpp"],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-m64",
            "-fopenmp",
            "-D NOISE_SEGMENT",
        ],
        libraries=["m", "gomp"],
        extra_objects=[
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgsl.a",
            "/data/project/ei_development/tools/gsl_build_shared/lib/libgslcblas.a",
        ],
        include_dirs=[
            "/data/project/ei_development/tools/gsl_build_shared/include",
            np.get_include(),
        ],
    )

setup(
    ext_modules=[bnm_ext],
    cmdclass={
        'build_ext': build_ext_w_cuda,
    },
)