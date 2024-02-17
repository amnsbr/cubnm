#ifndef UTILS_CUH
#define UTILS_CUH
#include "utils.hpp"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line);

__global__ void float2double(double **dst, float **src, size_t rows, size_t cols);
cudaDeviceProp get_device_prop(int verbose = 1);
#endif