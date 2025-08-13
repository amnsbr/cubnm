#define EXP exp
#define POW pow
#define SQRT sqrt
#define SIN sin
#define FMOD fmod
#define CUDA_MAX max
#define CUDA_MIN min

#define PI 3.14159265358979323846

#ifndef MAX_NODES_REG
    #define MAX_NODES_REG 200
#endif
#ifndef MAX_NODES_MANY
    #define MAX_NODES_MANY 1024
#endif
#ifndef MANY_NODES
    #define MAX_NODES MAX_NODES_REG
    #define __NOINLINE__ 
#else
    #define MAX_NODES MAX_NODES_MANY
    #define __NOINLINE__ __noinline__
    // in this case some __device__ kernels will be __noinline__ to avoid
    // using too much registers which would lead to 
    // "too many resources requested for launch"
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CUDA_CALLABLE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(GPU_ENABLED)
#define _GPU_ENABLED
// this is used to determine if the GPU-related
// declarations should be done, either if
// nvcc/hipcc is compiling the cuda source code
// or the C++ is being compiled but will be
// linked with the nvcc compiled library
// i.e., GPU_ENABLED is defined
#endif