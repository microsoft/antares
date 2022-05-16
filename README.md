## What is Antares:

**Antares** (https://github.com/microsoft/antares) is an engine to auto generate optimized kernels for [Multi Backends](backends). It is a framework not only for ***Software developers*** to get backend-related code, but also for ***Hardware developers*** to extend new backends/hareware quickly and easily. Antares frontend is based on [Antares IR](AntaresIR.md) that follows "One Language Syntax for All Platforms".

### How to Install:

```sh
python3 -m pip install --upgrade antares
```

### Quick Test:

```sh
BACKEND=c-scpu antares

# List Supported Backends
antares backends

# Help Information:
antares help
```

### Usage Examples (antares save/eval/compile):

```sh
# Quickly generate a multi-threaded CPU code:
BACKEND=c-mcpu antares

# Search an efficient multi-threaded CPU code and save best code to specified location:
STEP=100 BACKEND=c-mcpu antares save ./kernel_example.cpp

# Reproduce kernel evaluation based on an early saved source code:
BACKEND=c-mcpu antares eval ./kernel_example.cpp

# Freeze kernels and compiled into edge-side binaries:
BACKEND=c-mcpu antares compile ./kernel_example.cpp ./output-dest/
# Build solution in destination directory:
cd ./output-dest && make
```

### Advanced Examples:
```sh
# Quickly generate a CUDA code with correctness checking:
CHECK=1 BACKEND=c-cuda antares

# Search an efficient multi-threaded CPU code showing progress bar only:
PROGRESS=1 STEP=100 BACKEND=c-mcpu antares save ./kernel_example.cpp

# Quickly generate a SHADER code for Windows 10/11's DirectX12:
BACKEND=c-hlsl_win64 antares

# Quickly generate an ROCm code for AMDGPU (requires ROCm SDK >= 4.2):
BACKEND=c-rocm antares

# Quickly generate a CUDA code for computing MatMul (512,512)x(512,512) based on [Antares IR](AntaresIR.md) for NVIDIA GPU (requires NVIDIA CUDA SDK >= 10.0):
BACKEND=c-cuda COMPUTE_V1='- S = 512; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

# Search an efficient CUDA code for MatMul, using 2000 steps for trial:
BACKEND=c-cuda STEP=2000 COMPUTE_V1='- S = 512; einstein_v2(input_dict={"input0": {"dtype": "float32", "shape": [S, S]}, "input1": {"dtype": "float32", "shape": [S, S]}}, exprss="output0[N, M] +=! input0[N, K] * input1[K, M]")' antares

# Cleanup history caches:
antares clean

# Boot HTTP daemon for accepting searching tasks:
antares rest-server

# Setup Plugin for Pytorch && Examples:
BACKEND=c-cuda antares torch-setup
BACKEND=c-mcpu antares torch-setup
python3 -m antares_core.frameworks.pytorch.examples.1_hello_world
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

