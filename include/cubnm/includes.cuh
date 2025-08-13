#ifndef INCLUDES_CUH
#define INCLUDES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <map>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_vector_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <thread>
#include <chrono>
#include <iomanip>
#include <stdexcept>

#ifndef ENABLE_HIP
    // compiling with nvcc for Nvidia GPUs
    #include <cuda_runtime.h>
    #include <cooperative_groups.h>
#else
    // complinig with hipcc for AMD(/Nvidia) GPUs
    #include <hip_runtime.h>
    #include <hip_cooperative_groups.h>
    #include "cuda_to_hip.h"
#endif

namespace cg = cooperative_groups;

#endif