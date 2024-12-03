## Installation:

#### Quick Installation of AutoRT:

| Platform | OS Requirement | Python Requirement | Download Link |
| --- | --- | --- | --- |
| DirectX 12 (x86_64) | Windows >= 10 / Microsoft XBox | [Python3.12](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe) (Windows) | python3.12 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3+directx.win-cp312-cp312-win_amd64.whl |
| Vulkan 1.3 (x86_64) | Ubuntu >= 18.04  | Python3.12 (Linux) | python3.12 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3+vulkan.linux-cp312-cp312-manylinux1_x86_64.whl |
| CUDA >= 11.0 (x86_64) | Windows >= 10 / Ubuntu >= 20.04 | Python 3.10/3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3.6+cuda.zip |
| CUDA >= 12.0 (aarch64) | Ubuntu >= 22.04 | Python 3.10/3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3.6+cuda.zip |
| ROCm >= 6.2 (x86_64) | Ubuntu >= 20.04 | Python 3.10/3.12 | python3 -m pip install -U --no-build-isolation https://github.com/microsoft/antares/releases/download/v0.9.6/autort-0.9.6.3.6+cuda.zip |
| .. | .. | .. | .. (More coming soon) .. |

#### For CUDA, here are several container options for CUDA >= 11.0:

 * **Docker Image:** nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
 * **Docker Image:** nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
 * **Docker Image:** nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
 * **Docker Image:** nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04
 * ..

