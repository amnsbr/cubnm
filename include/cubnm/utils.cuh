#ifndef UTILS_CUH
#define UTILS_CUH
#include "utils.hpp"

inline void checkLast(const char* const file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDA Error (Kernel Launch) at " + std::string(file) + ":" +
            std::to_string(line) + "\n" + 
            cudaGetErrorName(err) + ": " + cudaGetErrorString(err)
        ));
    }
}
inline void checkReturn(cudaError_t value, const char* const file, const int line) {
    if (value != cudaSuccess) {
        throw std::runtime_error(std::string(
            "CUDA Error (API Call) at " + std::string(file) + ":" +
            std::to_string(line) + "\n" + 
            cudaGetErrorName(value) + ": " + cudaGetErrorString(value)
        ));
    }
}
#define CUDA_CHECK_RETURN(value) checkReturn(value, __FILE__, __LINE__)
#define CUDA_CHECK_LAST_ERROR() checkLast(__FILE__, __LINE__)

cudaDeviceProp get_device_prop(int verbose = 1);
#endif