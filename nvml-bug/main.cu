// System
#include <iostream>
#include <vector>

// Third-party
#include <cuda_runtime_api.h>

// Application
#include "cuda_utils.cuh"
#include "Nvml.h"

bool isInteger(char* str)
{
    while (*str)
    {
        if (!isdigit(*str++))
        {
            return false;
        }
    }
    return true;
}

__device__ int d_result;
__global__ void loop()
{
    volatile int infinity = 1;
    while (infinity)
    {
		infinity++;
    }
	d_result = infinity;
}

int main(int argc, char** argv)
{
    // Parse device ID from command line
    if (argc != 2 || !isInteger(argv[1]))
    {
        std::cout << "Usage: " << argv[0] << " DEVICE_ID" << std::endl;
        return 1;
    }
    const int device_id = atoi(argv[1]);

    // Get current NVIDIA driver version
    const auto nvml = new Nvml();
    std::cout << "NVIDIA driver version: " << nvml->GetDriverVersion() << std::endl;

    // Get device name
    cudaDeviceProp device_prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
    std::cout << "Using device ID " << device_id << " (" << device_prop.name << ")" << std::endl;

    // Check whether device is already in use by another instance of this application
    const std::string app_name(argv[0]);
    if (nvml->IsDeviceInUse(device_id, app_name))
    {
        std::cout << "ERROR: Device already in use" << std::endl;
        return 1;
    }

    // Start compute process on device
    std::cout << "Starting compute process on device (press CTRL+C to quit)..." << std::endl;
	CHECK_CUDA_ERROR(cudaSetDevice(device_id));
    loop<<<1, 1>>>();
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::cout << "Exiting";
    return 0;
}
