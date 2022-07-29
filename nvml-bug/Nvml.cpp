// Class
#include "Nvml.h"

// System
#include <algorithm>
#include <iostream>
#include <vector>
#define WIN32_LEAN_AND_MEAN 
#include <Windows.h>
#undef WIN32_LEAN_AND_MEAN

// Third-party
#include <cuda_runtime.h>
#include <nvml.h>

// Application
#include "cuda_utils.cuh"

#define CHECK_NVML_ERROR(nvmlError) { checkNvmlErrorFunc(nvmlError, __FUNCTION__, __FILE__, __LINE__); }

inline void checkNvmlErrorFunc(const nvmlReturn_t nvmlError, const char* const function, const char* const file, const int line)
{
    if (nvmlError != NVML_SUCCESS)
    {
        std::cout << nvmlErrorString(nvmlError) << std::endl;
        std::cout << "   at " << function << " in " << file << ":line " << line << std::endl;
        exit(nvmlError);
    }
}

Nvml::Nvml()
{
    // Load NVML DLL (delay loaded to ensure application will start even when NVML not present)
    // NOTE: At some point between versions 442.50 and 445.75, the NVIDIA Driver started to
    //       install NVML and nvidia-smi to "C:\Windows\System32" (which is on the system PATH
    //       by default) rather than "C:\Program Files\NVIDIA Corporation\NVSMI".
    //       Therefore, first try to load NVML DLL from system PATH (as this will be the most
    //       recent version), and then fall back to previous location if not found.
    _nvmlDll = LoadLibrary("nvml.dll");
    if (!_nvmlDll)
    {
        _nvmlDll = LoadLibrary(R"(C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll)");
    }

    if (!_nvmlDll)
    {
        std::cout << "ERROR: Unable to load NVML DLL" << std::endl;
        exit(1);
    }

    // Initialize NVML (must be called before any other NVML function)
    if (_nvmlDll)
    {
        nvmlInit();
    }
}

Nvml::~Nvml()
{
    // Shutdown NVML and free all resources
    if (_nvmlDll)
    {
        nvmlShutdown();
    }

    // Unload NVML DLL
    FreeLibrary(_nvmlDll);
}

std::string Nvml::GetDriverVersion() const
{
    // Get NVIDIA driver version
    char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    CHECK_NVML_ERROR(nvmlSystemGetDriverVersion(version, sizeof version));
    return version;
}

bool Nvml::IsDeviceInUse(const int deviceId, const std::string& appName) const
{
    // Get the device's PCI bus ID
    // NOTE: Must use PCI bus ID to identify device rather than CUDA device ID, since NVML ID is
    //       not guaranteed to correlate with other APIs; see:
    //       http://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g52677ecb45f937c5005124608780a3f4
    char pci_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    CHECK_CUDA_ERROR(cudaDeviceGetPCIBusId(pci_bus_id, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, deviceId));

    // Get NVML handle for the device
    nvmlDevice_t device;
    CHECK_NVML_ERROR(nvmlDeviceGetHandleByPciBusId(pci_bus_id, &device));

    // Get the number of processes with a compute context on the specified device
    // NOTE: This call returns NVML_ERROR_INSUFFICIENT_SIZE by design; see:
    //       http://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1g46ceaea624d5c96e098e03c453419d68
    unsigned int info_count = 0;
    auto nvml_error = nvmlDeviceGetComputeRunningProcesses(device, &info_count, nullptr);
    if (nvml_error != NVML_ERROR_INSUFFICIENT_SIZE)
    {
        CHECK_NVML_ERROR(nvml_error);
    }

    // Get information about each process with a compute context on the specified device
    // NOTE: This will fail with NVML_ERROR_NOT_SUPPORTED error on older GTX GPUs; see:
    //       https://developer.nvidia.com/nvidia_bug/2022059
    std::vector<nvmlProcessInfo_t> process_infos(info_count);
    CHECK_NVML_ERROR(nvmlDeviceGetComputeRunningProcesses(device, &info_count, process_infos.data()));

    // Convert the application name to lowercase
    std::string app_name(appName);
    transform(app_name.begin(), app_name.end(), app_name.begin(), tolower);

    // Check whether another process from the specified application is currently using the specified device
    const auto current_process_id = GetCurrentProcessId();
    for (const auto& process_info : process_infos)
    {
        // Ignore the current process
        // NOTE: This is necessary since the call to cudaDeviceGetPCIBusId() above will initialize
        //       a compute context on the specified device for the current process
        if (process_info.pid == current_process_id)
        {
            continue;
        }

        // Get the process name
        // NOTE: This can fail with "Insufficient Permissions" for some processes
        constexpr int process_name_len = 512;
        std::string process_name(process_name_len, ' ');
        nvml_error = nvmlSystemGetProcessName(process_info.pid, &process_name[0], process_name_len);
        if (nvml_error == NVML_ERROR_NO_PERMISSION)
        {
            continue;
        }
        CHECK_NVML_ERROR(nvml_error);

        // Trim the process name to remove any characters after the null terminator
        // NOTE: This is necessary otherwise the equality check will fail
        process_name = process_name.c_str();

        // Convert the process name to lowercase
        transform(process_name.begin(), process_name.end(), process_name.begin(), tolower);

        // Check whether the process is another application process by comparing the end of the name
        if (equal(app_name.rbegin(), app_name.rend(), process_name.rbegin()))
        {
            return true;
        }
    }

    return false;
}
