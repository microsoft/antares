## AutoRT: the Next Generation of Antares.

***Path 1 (Antares for Kernel Optimization):*** Blackbox Code Optimizer (CUDA/ROCm/DX/SYCL/OCL/CPU/IPU/Android):

&nbsp;&nbsp;&nbsp;&nbsp;`python3 -m pip install antares`, which follows: [README for Antares](README-legacy.md).

***Path 2 (AutoRT for Runtime):*** Pytorch Runtime & Benchmark based on Antares Drivers (DirectX/Vulkan/CUDA/CPU/..):

AutoRT is a compiler solution that helps runtime users to invent, benchmark and optimize operators for Pytorch using your own accelerators:
- AutoRT can be as a [benchmark utility](#--playground-1---benchmark-your-windows-device) for device performance testing and profiling.
- AutoRT can also generate Pytorch2 of your device to accelerate standard [Pytorch applications](#quick-test-2-mnist-training-by-pytorch2-using-windows-directx) (e.g. DirectX).
- Additionally, AutoRT futher helps to construct [custom defined](#quick-test-1-create-custom-operator-of-your-own-in-pytorch-2) / fused operators that are beyond the built-in functions of Pytorch.
- ***AutoRT for Windows DirectX 12 / Linux CUDA*** has experimental version [released](#--quick-installation-of-autort).
- Click [here](https://github.com/microsoft/antares/issues/new) to suggest more platforms (e.g. Pytorch2 for Windows ROCm / OpenCL / SYCL / Apple Metal / ..) you would like AutoRT to support in the follow-up releases.

#### Archtecture of AutoRT as a Backend for Pytorch 2.0:
<p align="center">
  <img src="AutoRT4Torch.svg" data-canonical-src="AutoRT4Torch.svg" width="650" height="230" />
</p>

#### Workflow of Custom Operations from Antares IR to Different Backends:
<p align="center">
  <img src="AutoRT-opt.svg" data-canonical-src="AutoRT-opt.svg" width="650" height="120" />
</p>


## - Quick Installation of AutoRT:

#### Installation

| Platform | OS Requirement | Python Requirement | Download Link |
| --- | --- | --- | --- |
| DirectX 12 | Windows >= 10 / Microsoft XBox | [Python3.12](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe) (Windows) | python3.12 -m pip install -r https://github.com/microsoft/antares/releases/download/v0.9.3/autort_for_dxwin.py312 |
| Vulkan 1.3 | Ubuntu >= 18.04 (or images)  | [Python3.12](https://github.com/ghostplant/collections/releases/download/utilities/python-3.12-linux-x86_64.deb) (Linux) | python3.12 -m pip install -r https://github.com/microsoft/antares/releases/download/v0.9.3/autort_for_vklinux.py312 |
| CUDA >= 11 | Ubuntu >= 18.04 (or images) | Python 3.8/3.9/3.10/3.11/3.12 | python3 -m pip install -r https://github.com/microsoft/antares/releases/download/v0.9.3/autort_for_cuda_linux.py3x |
| .. | .. | .. | .. (More coming soon) .. |

For CUDA, here are several Ubuntu >= 18.04 equivalent containers below:
 * **Docker Image:** nvidia/cuda:12.0.1-cudnn8-devel-ubuntu18.04
 * **Docker Image:** nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
 * **Docker Image:** nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04
 * **Docker Image:** nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
 * ..

## - Playground 1 - Benchmark your Windows Device:

#### Quick Test 1: Benchmark to evaluate device memory bandwidth over DirectX 12.
```sh
$ python.exe -m autort.utils.memtest
  ...
  [1000/1000] AutoRT Device Memory Bandwidth: (Actual ~= 468.12 GB/s) (Theoretical ~= 561.75 GB/s)
```

#### Quick Test 2: Benchmark to evaluate device FP32 performance over DirectX 12.
```sh
$ python.exe -m autort.utils.fp32test
  ...
  [5000/5000] AutoRT FP32 TFLOPS: (Actual ~= 9.84 TFLOPS) (Theoretical ~= 10.93 TFLOPS)
```

## - Playground 2 - Running Pytorch2 over DirectX:

#### Quick Test 1: Create "custom operator" of your own in Pytorch 2.

- **Style-1: "AutoRT API Style"** Custom Operator Generation:
```py
>> import torch, autort
>> data = torch.arange(0, 10, dtype=torch.float32, device=autort.device())

>> f = autort.export(ir="sigmoid_f32[N] = 1 - 1 / (1 + data[N].call(strs.exp))", inputs=["data=float32[N:4096000]"], config="tune:5")
>> print(f(data))
tensor([0.5000, 0.7311, 0.8808, 0.9526, 0.9820, 0.9933, 0.9975, 0.9991, 0.9997, 0.9999])
>> print(autort.ops.sigmoid_f32(data))
tensor([0.5000, 0.7311, 0.8808, 0.9526, 0.9820, 0.9933, 0.9975, 0.9991, 0.9997, 0.9999])
```

- **Style-2: "Command Line Style"** Custom Operator Generation:
```sh
# Fist, create a custom sigmoid activation operator with auto-tuning steps == 10:
$ autort --ir "sigmoid_f32[N] = 1 - 1 / (1 + data[N].call(strs.exp))" -i data=float32[N:4096000] -c "tune:5"

# Then, use it in Pytorch 2 session:
$ python.exe
>> import torch, autort
>>
>> data = torch.arange(0, 10, dtype=torch.float32, device=autort.device())
>> output = autort.ops.sigmoid_f32(data)
>> print(output)
tensor([0.5000, 0.7311, 0.8808, 0.9526, 0.9820, 0.9933, 0.9975, 0.9991, 0.9997,
        0.9999])
```


#### Quick Test 2: MNIST Training by Pytorch2 (DirectX only):
```sh
$ python.exe -m autort.examples.02_mnist
  ...
  step = 100, loss = 2.2871, accuracy = 21.88 %
  step = 200, loss = 2.1408, accuracy = 46.88 %
  step = 300, loss = 1.6713, accuracy = 62.50 %
  step = 400, loss = 0.9573, accuracy = 62.50 %
  step = 500, loss = 0.8338, accuracy = 68.75 %
  step = 600, loss = 0.5882, accuracy = 84.38 %
  step = 700, loss = 0.2738, accuracy = 87.50 %
  step = 800, loss = 0.5159, accuracy = 87.50 %
  step = 900, loss = 0.5511, accuracy = 84.38 %
  step = 1000, loss = 0.2616, accuracy = 93.75 %
  ...
```

#### Quick Test 3: Fine-tune existing operators to make Pytorch Builtin Operators run faster (DirectX only).
```sh
$ python.exe -m autort.utils.mmtest

  >> Performance of your device:

     `MM-Perf` (current) = 4.15 TFLOPS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  >> ...

$ python -m autort.utils.export -s 4000

  Module file for operator `gemm_f32` has been exported to `.\ops\gemm_f32.mod`.

  ..

$ python.exe -m autort.utils.mmtest

  >> Performance of your device:

     `MM-Perf` (current) = 9.71 TFLOPS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  >> ...
```

If you like it, welcome to report issues or donate stars which can encourage AutoRT to support more backends, more OS-type and more documentations. See More Information about Microsoft [Contributing](CONTRIBUTING.md) and [Trademarks](TRADEMARKS.md).
