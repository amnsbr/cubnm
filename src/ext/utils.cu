#include "cubnm/includes.cuh"
#include "cubnm/defines.h"
#include "cubnm/utils.cuh"

cudaDeviceProp get_device_prop(int verbose) {
    /*
        Gets GPU device properties and prints them to the console.
        Also throws an error if no GPU is found.
    */
    int count;
    cudaError_t error = cudaGetDeviceCount(&count);
    struct cudaDeviceProp prop;
    if (error == cudaSuccess) {
        int device = 0;
        CUDA_CHECK_RETURN( cudaGetDeviceProperties(&prop, device) );
        if (verbose > 0) {
            std::cout << std::endl << "CUDA device #" << device << ": " << prop.name << std::endl;
        }
        if (verbose > 1) {
            std::cout << "totalGlobalMem: " << prop.totalGlobalMem << ", sharedMemPerBlock: " << prop.sharedMemPerBlock; 
            std::cout << ", regsPerBlock: " << prop.regsPerBlock << ", warpSize: " << prop.warpSize << ", memPitch: " << prop.memPitch << std::endl;
            std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << ", maxThreadsDim[0]: " << prop.maxThreadsDim[0] 
                << ", maxThreadsDim[1]: " << prop.maxThreadsDim[1] << ", maxThreadsDim[2]: " << prop.maxThreadsDim[2] << std::endl; 
            std::cout << "maxGridSize[0]: " << prop.maxGridSize[0] << ", maxGridSize[1]: " << prop.maxGridSize[1] << ", maxGridSize[2]: " 
                << prop.maxGridSize[2] << ", totalConstMem: " << prop.totalConstMem << std::endl;
            std::cout << "major: " << prop.major << ", minor: " << prop.minor << ", clockRate: " << prop.clockRate << ", textureAlignment: " 
                << prop.textureAlignment << ", deviceOverlap: " << prop.deviceOverlap << ", multiProcessorCount: " << prop.multiProcessorCount << std::endl; 
            std::cout << "kernelExecTimeoutEnabled: " << prop.kernelExecTimeoutEnabled << ", integrated: " << prop.integrated  
                << ", canMapHostMemory: " << prop.canMapHostMemory << ", computeMode: " << prop.computeMode <<  std::endl; 
            std::cout << "concurrentKernels: " << prop.concurrentKernels << ", ECCEnabled: " << prop.ECCEnabled << ", pciBusID: " << prop.pciBusID
                << ", pciDeviceID: " << prop.pciDeviceID << ", tccDriver: " << prop.tccDriver  << std::endl;
        }
    } else {
        throw std::runtime_error(std::string("No CUDA devices was found"));
    }
    return prop;
}

bool is_running_on_wsl() {
    /* 
    Checks whether on runtime the program is being run
    on WSL.
    This is needed to skip some parts of the program
    which are not supported by the Nvidia driver for
    Windows.
    */
    // looking for microsoft or WSL in /proc/version
    std::ifstream procVersion("/proc/version");
    if (procVersion.is_open()) {
        std::string line;
        std::getline(procVersion, line);
        procVersion.close();
        
        if (line.find("Microsoft") != std::string::npos || 
            line.find("microsoft") != std::string::npos ||
            line.find("WSL") != std::string::npos) {
            return true;
        }
    }

    // look for WSLInterop file
    std::ifstream wslInterop("/proc/sys/fs/binfmt_misc/WSLInterop");
    if (wslInterop.good()) {
        return true;
    }
    
    // check environment variables
    const char* wslDistroName = std::getenv("WSL_DISTRO_NAME");
    const char* wslInteropEnv = std::getenv("WSL_INTEROP");
    
    return (wslDistroName != nullptr || wslInteropEnv != nullptr);
}