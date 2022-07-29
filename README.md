# nvml-bug

## Overview

This sample code demonstrates an issue with recent versions of NVIDIA's NVML library whereby the
NVML function to get the list of compute processes that are currently running on a given GPU device
(`nvmlDeviceGetComputeRunningProcesses`) now seems to return compute processes across **all** GPU
devices (rather than just the specified GPU device, as was the case for previous versions of NVML).

The NVML library is included in the NVIDIA GPU driver installation. It appears that any version `>= 500`
(e.g. v5xx) is affected. NVIDIA GPU driver versions `< 500` (e.g. v4xx) are unaffected (i.e. the
`nvmlDeviceGetComputeRunningProcesses` function returns compute processes for only the specified GPU
device). This has been tested across a range of different GeForce/Quadro/Tesla GPU devices. Whilst the
code here uses CUDA Toolkit 11.3, the issue can also be reproduced using the latest CUDA Toolkit 11.7.

## Requirements

* Windows 10 (or later)
* Visual Studio 2017
* CUDA Toolkit 11.3
* NVIDIA GPU driver v4xx
* NVIDIA GPU driver v5xx
* At least 2 CUDA-enabled GPU devices

## Run

Usage: `nvml-bug.exe <device_id>`

## Reproduce Issue

To validate the system, first make sure that a NVIDIA GPU driver v4xx is installed and then open a Command Prompt and run:

    nvml-bug.exe 0

Once "Starting compute process on device (press CTRL+C to quit)..." is output to the console window,
open a second command prompt and run:

    nvml-bug.exe 1

This should output no error messages. Now close or stop both applications.

To reproduce the issue, make sure that a NVIDIA GPU driver v5xx is installed and then re-run the
procedure listed above. The second invocation of the application (i.e. `nvml-bug.exe 1`) should now
erroneously fail with a "ERROR: Device already in use" error message.

## Example Output

### NVIDIA GPU Driver v4xx

Observed (correct) output when running application on separate GPU devices:

```
NVIDIA driver version: 473.47
Using device ID 0 (NVIDIA RTX A4000)
Starting compute process on device (press CTRL+C to quit)...
```

```
NVIDIA driver version: 473.47
Using device ID 1 (Quadro P400)
Starting compute process on device (press CTRL+C to quit)...
```

Observed (correct) output when running application on the same GPU device:

```
NVIDIA driver version: 473.47
Using device ID 0 (NVIDIA RTX A4000)
Starting compute process on device (press CTRL+C to quit)...
```

```
NVIDIA driver version: 473.47
Using device ID 0 (NVIDIA RTX A4000)
ERROR: Device already in use
```

### NVIDIA GPU Driver v5xx

Observed (**incorrect**) output when running application on separate GPU devices:

```
NVIDIA driver version: 516.59
Using device ID 0 (NVIDIA RTX A4000)
Starting compute process on device (press CTRL+C to quit)...
```

```
NVIDIA driver version: 516.59
Using device ID 1 (Quadro P400)
ERROR: Device already in use
```

Observed (correct) output when running application on the same GPU device:

```
NVIDIA driver version: 516.59
Using device ID 0 (NVIDIA RTX A4000)
Starting compute process on device (press CTRL+C to quit)...
```

```
NVIDIA driver version: 516.59
Using device ID 0 (NVIDIA RTX A4000)
ERROR: Device already in use
```
