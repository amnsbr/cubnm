from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.spawn import find_executable
import subprocess
import os
import fnmatch
import sys
import numpy as np
import versioneer

ROOT = os.path.abspath(os.path.dirname(__file__))

def find_file(file_name, search_path):
    result = []
    for root, dirs, files in os.walk(search_path):
        for name in fnmatch.filter(files, file_name):
            result.append(os.path.join(root, name))
    return result

def get_gpu_model(method='compiler'):
    """
    Detects the GPU type (if any) to compile for.
    It first checks for Nvidia compiler/smi and then
    ROCm.

    Parameters
    ----------
    method: {'compiler', 'smi'}
    
    Returns
    -------
    {False, 'nvidia', 'rocm'}
    """
    if method == 'smi':
        # nvidia
        try:
            output = subprocess.check_output(['nvidia-smi', '-L'])
            output = output.decode('utf-8').strip()
            gpu_list = output.split('\n')
            gpu_count = len(gpu_list)
            if gpu_count>0:
                return 'nvidia'
        except:
            pass
        # rocm
        try:
            output = subprocess.check_output(['rocm-smi', '-i', '--json'])
            output = output.decode('utf-8').strip()
            gpu_dict = json.loads(output)
            gpu_count = len(gpu_dict)
            if gpu_count>0:
                return 'rocm'
        except:
            pass
        return False
    elif method == 'compiler':
        # can be useful for compiling code on a
        # non-GPU system for running it later
        # on GPUs
        # nvidia
        if find_executable('nvcc'):
            return 'nvidia'
        elif find_executable('hipcc'):
            return 'rocm'
        else:
            return False

# specify installation options
many_nodes = not ("CUBNM_NO_MANY_NODES" in os.environ)
max_nodes_reg = os.environ.get("CUBNM_MAX_NODES_REG", 200)
max_nodes_many = os.environ.get("CUBNM_MAX_NODES_MANY", 1024)
gpu_model = get_gpu_model(method='compiler')
gpu_enabled = (gpu_model != False)
omp_enabled = not ("CUBNM_NO_OMP" in os.environ)
noise_segment = not ("CUBNM_NOISE_WHOLE" in os.environ)

# Write the flags to a temporary _flags.py file
with open(os.path.join(ROOT, "src", "cubnm", "_setup_opts.py"), "w") as flag_file:
    flag_file.write(
        "\n".join(
            [f"many_nodes_flag = {many_nodes}", 
             f"gpu_enabled_flag = {gpu_enabled}",
             f"omp_enabled_flag = {omp_enabled}",
             f"noise_segment_flag = {noise_segment}",
             f"max_nodes_reg = {max_nodes_reg}",
             f"max_nodes_many = {max_nodes_many}",
             f"gpu_model_flag = '{gpu_model}'"
            ]
        )
    )

# determine libraries shared between GPU and CPU versions
libraries = ["m", "rt"]
if omp_enabled:
    libraries.append("gomp")

_CC = os.environ.get("CC", "gcc") # this will be used for compiling GSL + restoring os $CC and $CXX at the end
_CXX = os.environ.get("CXX", "g++")
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# create lists of include directories
shared_includes = [
    os.path.join(ROOT,"include"),
    os.path.join(ROOT, "src", "ext"),
    np.get_include(),
    "/opt/miniconda/include" # added for conda-based cibuildwheel
]
if gpu_model == 'nvidia':
    gpu_includes = [
        "/usr/lib/cuda/include",
        "/usr/include/cuda",
    ]
elif gpu_model == 'rocm':
    ROCM_PATH = subprocess.check_output(['hipconfig', '--path']).decode('utf-8').strip()
    gpu_includes = [os.path.join(ROCM_PATH,"include", "hip")]
else:
    gpu_includes = []
all_includes = shared_includes + gpu_includes

# extra compile args shared between CPU and GPU
extra_compile_args = [
    "-std=c++11",
    "-O3",
    "-m64",
    f"-D MAX_NODES_REG={max_nodes_reg}",
    f"-D MAX_NODES_MANY={max_nodes_many}"
]
if noise_segment:
    extra_compile_args.append("-D NOISE_SEGMENT")
if many_nodes:
    extra_compile_args.append("-D MANY_NODES")
if omp_enabled:
    extra_compile_args += [
        "-fopenmp",
        "-D OMP_ENABLED"
    ]
if gpu_enabled:
    extra_compile_args.append("-D GPU_ENABLED")
    if gpu_model == 'rocm':
        extra_compile_args.append("-D ENABLE_HIP")

if gpu_enabled:
    print(f"Compiling for GPU ({gpu_model}) + CPU")
    libraries += ["bnm"]
    if gpu_model == 'nvidia':
        libraries += ["cudart_static"]
        library_dirs = [
            "/usr/lib/cuda", 
            "/usr/local/cuda/lib64",
        ]
    elif gpu_model == 'rocm':
        # TODO: is there an alternative to `cudart_static`?
        libraries += ['amdhip64']
        library_dirs = [
            os.path.join(ROCM_PATH, 'lib'),
        ]
    bnm_ext = Extension(
        "cubnm.core",
        [os.path.join("src", "ext", "core.cpp")],
        language="c++",
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        include_dirs=all_includes,
        library_dirs=library_dirs + \
            [
                os.path.join(ROOT, "src", "ext"),
                "/opt/miniconda/lib" # added for conda-based cibuildwheel
            ] + \
            os.environ.get("LIBRARY_PATH","").split(":") + \
            os.environ.get("LD_LIBRARY_PATH","").split(":")
        ,
    )
else:
    print("Compiling for CPU")
    bnm_ext = Extension(
        "cubnm.core",
        [os.path.join("src", "ext", "core.cpp")],
        language="c++",
        extra_compile_args=extra_compile_args,
        libraries=libraries,
        include_dirs=shared_includes,
        library_dirs=[
            "/opt/miniconda/lib" # added for conda-based cibuildwheel
        ] + \
        os.environ.get("LIBRARY_PATH","").split(":") + \
        os.environ.get("LD_LIBRARY_PATH","").split(":"),
    )

# extend build_ext to also build GSL (if needed) and compile GPU code
class build_ext_gsl_cuda(build_ext):
    def build_extensions(self):
        # Build GSL (if needed)
        # search for libgsl.a and libgslcblas.a in some common paths
        lib_dirs = os.environ.get("LIBRARY_PATH","").split(":") + \
                os.environ.get("LD_LIBRARY_PATH","").split(":") + \
                [
                    "/usr/lib",
                    "/lib",
                    "/usr/local/lib",
                    "/opt/miniconda/lib", # cibuildwheel
                    # TODO: identify and search current conda env
                    os.path.join(os.environ.get('HOME', '/opt'), '.cubnm', 'gsl', 'build', 'lib'), # has been installed before by cuBNM
                ]
        found_gsl = False
        for lib_dir in lib_dirs:
            if ((lib_dir!='') & os.path.exists(lib_dir)):
                r = find_file('libgsl.a', lib_dir)
                if r: # assuming libgsl.a and libgslcblas.a are in the same directory
                    found_gsl = True
                    GSL_LIB_DIR = os.path.dirname(r[0])
                    print(f"Found libgsl.a and libgslcblas.a in {GSL_LIB_DIR}")
                    break
        if not found_gsl:
            print("Downloading and building GSL")
            try:
                gsl_dir = os.path.join(os.environ.get('HOME', '/opt'), '.cubnm', 'gsl')
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
                f"curl https://mirror.ibcp.fr/pub/gnu/gsl/gsl-2.7.tar.gz -o {gsl_tar}",
                f"cd {gsl_dir} && tar -xf {gsl_tar} &&"
                f"cd {gsl_src} && ./configure --prefix={gsl_build} --enable-shared &&"
                f"make && make install",
            ]
            for command in gsl_setup_commands:
                result = subprocess.run(command, shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
                # print(f"Command: {command}")
                # print(f"Return Code: {result.returncode}")
                # print(f"stderr:\n{result.stderr}")
            GSL_LIB_DIR = os.path.join(gsl_build, 'lib')
        # Compile CUDA code into libbnm.a
        if gpu_enabled:
            cuda_dir = os.path.join(ROOT, 'src', 'ext')
            conf_flags = []
            if noise_segment:
                conf_flags.append("NOISE_SEGMENT")
            if many_nodes:
                conf_flags.append("MANY_NODES")
            if gpu_model == 'rocm':
                conf_flags.append("ENABLE_HIP")
                include_flags = " ".join([f"-I {p}" for p in all_includes])
            else:
                include_flags = " ".join([f"-I {p}" for p in shared_includes])
            conf_flags = " ".join([f"-D {f}" for f in conf_flags])
            source_files = ["bnm.cu", "utils.cu", "fc.cu"] + [
                os.path.relpath(f, start=cuda_dir) for f in find_file('*.cu', os.path.join(cuda_dir, 'models'))
            ]
            # create a unified source file including all .cu files
            # this offers significantly better performance than 
            # compiling each file separately and linking them later
            unified_source_path = os.path.join(cuda_dir, "_bnm.cu")
            with open(unified_source_path, 'w') as unified_file:
                for source_file in source_files:
                    unified_file.write(f'#include "{source_file}"\n')
            if gpu_model == 'nvidia':
                compile_commands = [
                    f"nvcc -c -rdc=true -std=c++11 --compiler-options '-fPIC' -o {cuda_dir}/_bnm.o {unified_source_path} "
                    f"{include_flags} {conf_flags}",
                    f"nvcc -dlink --compiler-options '-fPIC' -o {cuda_dir}/_bnm_linked.o {cuda_dir}/_bnm.o "
                        f"-L {GSL_LIB_DIR} -lm -lgsl -lgslcblas -lcudart_static",
                    f"ar cru {cuda_dir}/libbnm.a {cuda_dir}/_bnm_linked.o {cuda_dir}/_bnm.o",
                    f"ranlib {cuda_dir}/libbnm.a",
                ]
            else:
                compile_commands = [
                    f"hipcc -fno-gpu-rdc -std=c++11 -fPIC {include_flags} {conf_flags} -c {unified_source_path} -o {cuda_dir}/_bnm.o ",
                    f"hipcc --shared -fPIC -fno-gpu-rdc -o {cuda_dir}/_bnm_linked.so {cuda_dir}/_bnm.o "
                        f"-L {GSL_LIB_DIR} -lgsl -lgslcblas -L {ROCM_PATH}/lib -lamdhip64", # -lm and -lcudart_static are removed
                    f"ar cru {cuda_dir}/libbnm.a {cuda_dir}/_bnm_linked.so {cuda_dir}/_bnm.o",
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
    version=versioneer.get_version(),
    ext_modules=[bnm_ext],
    cmdclass=versioneer.get_cmdclass({
        'build_ext': build_ext_gsl_cuda,
    }),
)

# restore OS's original $CC and $CXX
os.environ['CC'] = _CC
os.environ['CXX'] = _CXX