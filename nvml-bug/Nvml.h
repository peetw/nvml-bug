#pragma once

// System
#include <string>

// Forward declarations
#ifndef _WINDEF_
struct HINSTANCE__; // Avoids need to include windows.h
typedef HINSTANCE__* HMODULE;
#endif

class Nvml
{
public:
    // Load DLL and initialize NVML
    Nvml();

    // Shutdown NVML and free all resources
    ~Nvml();

    // Get NVIDIA driver version
    std::string GetDriverVersion() const;

    // Check whether the specified device is already being used by another instance of this application
    bool IsDeviceInUse(const int deviceId, const std::string& appName) const;
private:
    HMODULE _nvmlDll;
};
