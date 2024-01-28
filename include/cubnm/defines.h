// Using floats vs doubles in the simulations, pass -D USE_FLOATS=1 to compiler to use floats
// Note that this does not affect FIC, CMAES and GOF calculations (always using doubles)
// as well as the noise array (always generated as floats, but then casted to u_real)
#ifdef USE_FLOATS
    typedef float u_real;
    #define EXP expf
    #define POW powf
    #define SQRT sqrtf
    #define CUDA_MAX fmaxf
    #define CUDA_MIN fminf
#else
    typedef double u_real;
    #define EXP exp
    #define POW pow
    #define SQRT sqrt
    #define CUDA_MAX max
    #define CUDA_MIN min
#endif

#ifndef MANY_NODES
    #define MAX_NODES 500
#else
    // this is just an arbitrary number
    // and it is not guaranteed that the code will work 
    // with this many nodes or won't work with more nodes
    #define MAX_NODES 10000
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
// #define CUDA_CONSTANT __constant__
#else
#define CUDA_CALLABLE_MEMBER
// #define CUDA_CONSTANT
#endif 