## Installation:

#### Quick Installation of AutoRT:

| Platform | OS Requirement | Python Requirement | Download Link |
| --- | --- | --- | --- |
| CUDA >= 12.x (x86_64) | Windows >= 10 / Ubuntu >= 20.04 | Python == 3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.6.0+cuda.zip |
| CUDA >= 12.x (aarch64) | Ubuntu >= 24.04 | Python == 3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.6.0+cuda.zip |
| ROCm == 7.2.4 (x86_64) | Ubuntu >= 24.04 | Python == 3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.6.0+cuda.zip |
| DirectX 12 (x86_64) | Windows >= 10 / Microsoft XBox | [Python3.12](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe) (Windows) | python3.12 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3+directx.win-cp312-cp312-win_amd64.whl |
| Vulkan 1.3 (x86_64) | Ubuntu >= 18.04  | Python3.12 (Linux) | python3.12 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3+vulkan.linux-cp312-cp312-manylinux1_x86_64.whl |
| .. | .. | .. | .. |

#### Recommended Docker Image:

 * **Docker Image for CUDA:** nvidia/cuda:12.8.0-devel-ubuntu24.04
 * **Docker Image for ROCM:** rocm/dev-ubuntu-24.04:7.2.4
 * ..

