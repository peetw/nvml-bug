#pragma once

// System
#include <iostream>

// Third-party
#include <cuda_runtime_api.h>

#define CHECK_CUDA_ERROR(cudaError) { checkCudaErrorFunc(cudaError, __FUNCTION__, __FILE__, __LINE__); }

inline void checkCudaErrorFunc(const cudaError_t cudaError, const char* const function, const char* const file, const int line)
{
    if (cudaError != cudaSuccess)
    {
        std::cout << cudaGetErrorString(cudaError) << std::endl;
        std::cout << "   at " << function << " in " << file << ":line " << line << std::endl;
        exit(cudaError);
    }
}
