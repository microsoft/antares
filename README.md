# What is Antares:
- Antares is an automatic engine for multi-platform kernel generation and optimization (targeting to CUDA/ROCm/CPU/DirectX12/Graphcore).
- Antares simplifies most TVM's low-level features, making it easier to use for DNN developers on Microsoft related platforms.
- Antares follows "_One Language Syntax for All Platforms_" principle to reduce the description complexity on different platforms.

## Documentation for Quick Start
1. [Quick Start to install Antares for DirectX12](platforms/c-hlsl/evaluator/AntaresEvalAgent)
2. [Quick Start to write A + B shaders for DirectX12 using CUDA-like interfaces](platforms/c-hlsl/evaluator/AntaresHlslLib/examples)
3. [Quick Start for Antares IR examples](AntaresIR.md)

# Antares Functionality:
- Antares can convert computing operators from your DNN models into low-level source codes of the target device (e.g. kernels, shaders, ..).
- Antares can also automatically tune and optimize these DNN operators on end-to-end device using efficient mechanisms and algorithms.

# Antares can especially help you on these cases:
- You want to modify fine-grain DNN workloads, but Tensorflow/Pytorch's built-in implementation are limited.
- You notice some operators are inefficent, and you want to replace it with a better one easily.
- Plus MSRA's NNfusion project, you can port your full DNN models into Window executable and get acceleration with DirectX12 + Intel/AMD/NVIDIA graphic cards.
- You want to split fine-grain operator workloads into the local tile node of Graphcore, which benifits the on-ship memory usage and reduces BSP communication overhead.
- Evaluate the compiler or potential runtime efficiency within Antares supported accelerators, e.g. A100.
- Antares provides a large domain for researchers to develop on kernel optimizations, e.g. custom tuners, custom schedule policies, custom platforms, etc.


# Install Antares:

```sh
sudo apt install docker.io

git clone https://github.com/microsoft/antares

cd antares/
sudo BACKEND=c-cuda make

# If you need Antares to extend/boost Tensorflow operators, please also run:
sudo python3 ./frameworks/antares/tensorflow/setup.py
# (Recommended Tensorflow CUDA Installation Source (for CUDA 10.0): pip3 install --upgrade pip && pip3 install tensorflow-gpu==1.15.3)

# If you need Antares to extend/boost Pytorch operators, please also run:
sudo python3 ./frameworks/antares/pytorch/setup.py
# (Recommended Pytorch CUDA Installation Source (for CUDA 10.0): pip3 install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html)
```

# Startup with First Example (CUDA example):

```sh
cd ${ANTARES_ROOT}/
sudo BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, M] +=! input0[N, K] * input1[K, M]", { "input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [512, 512]}})' make
# Other valid platforms for BACKEND variable could be: c-rocm, c-hlsl, c-gc, c-mcpu, c-ocl, ..
```

# Example with Tensorflow-GPU/Pytorch-GPU:

This example shows you an easy way to quickly add custom operators in Tensorflow/Pytorch, but the operator itself is not an optimized version (not tuned).

```sh
# First, launch the antares REST server (a CUDA example)

BACKEND=c-cuda make rest-server
```

- Tensorflow Frontend Only:
```py
# For Tensorflow CUDA frontend, just execute the following python script:

import tensorflow as tf
from tensorflow.contrib import antares

x = tf.random.uniform([1024, 512])

op = antares.make_op('reduce_sum_0[N] +=! data[N, M]', {'data': x})

with tf.Session() as sess:
  print('The result of tensor `%s` is:\n%s' % (op._output_names[0], sess.run(op)))

```

- Pytorch Frontend Only:
```py
# For Pytorch frontend, just execute the following python script:

import os, torch
from torch.contrib.antares.custom_op import CustomOp

device = torch.device("cuda")
dtype = torch.float32
custom_op = CustomOp().to(device, dtype)

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

x = torch.randn(1024, 512, **kwargs)

custom_op = CustomOp().to(device, dtype)

inputs = {'data': x}
outputs = custom_op('reduce_sum_0[N] +=! data[N, M]', values=list(inputs.values()), keys=list(inputs.keys()))
print('The result of tensor `%s` is:\n%s' % (custom_op._output_names[0], outputs))

```

If you want the operator you just extended to run more efficiently, you can consider to take a look at "How to Tune Expressions" sections below.

# Documentation for Other Advanced Examples:

For more syntax usage or examples, please follow documentation here: [Antares IR & Examples](AntaresIR.md)

Antares can support multi-line statements as long as they are fuse-able, for example of ConvReluBias:

```
    conv_out[N, F, HO, WO] +=! input_data[N, C, HO + KH, WO + KW] * kernel[KH, KW, C, F] where HO in 256, WO in 256;

    conv_bias[N, F, HO, WO] = conv_out[N, F, HO, WO] + bias[0, F, 0, 0];

    output0[N, F, HO, WO] = conv_bias[N, F, HO, WO].when(conv_bias[N, F, HO, WO] > 0.0, 0.0);
```

# Antares Additional Features (comparing to TVM):

|   | Antares | TVM |
|---|---|---|
| Platform: DirectX12 | Y | - |
| Platform: ROCm HIP C |  Y | - |
| Platform: GraphCore | Y | - |
| Decoupling for Multi-Platforms | Y | - |
| Workflow: Auto Shard | Y | - |
| Workflow: Auto Infershape | Y | - |
| Language | Antares IR | Hyrbid Script/Topi/.. |
| Framework: JIT Op Maker for Tensorflow | Y | - |
| Framework: JIT Op Maker for Pytorch | Y | - |

# Current Feature Table:

|       | HIP-C(c-rocm) | CUDA(c-cuda) | CPU(c-mcpu) | DirectX12(c-hlsl) | Graphcore(c-gc) | (..coming soon..) |
|---|---|---|---|---|---|---|
| Global schedules | Y | Y | Y | Y | Y |  |
| Local schedules | Y | Y | Y | Y |  |  |
| Head fusion | Y | Y | Y | Y | Y |  |
| Tail fusion | Y | Y |  | Y |  |  |
| Evaluator | Y | Y | Y | Y |  |  |
| Tensorflow Plugin | Y | Y |  |  |  |  |
| Pytorch Plugin | Y | Y |  |  |  |  |
| NNfusion Plugin | Y | Y | Y | Y | Y |  |
| Blend Intricsic | Y | Y | Y | Y |  |  |

-----------

# How to Tune Expressions:

## Local tuning:

If you want automatic ways to optimize the operator (described in your environmental variable `COMPUTE_V1`), you just need to add one more variable in your first-run examples: `STEP=1000`,
which means Antares will take 1000 chances to search for a potenially better kernel version. For example,

```sh
    STEP=1000 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * 4 + KH, WO * 4 + KW] * input1[F, C, KH, KW] where HO in 55, WO in 55", input_dict={"input0": {"dtype": "float32", "shape": [64, 3, 227, 227]}, "input1": {"dtype": "float32", "shape": [96, 3, 11, 11]}});' make
```

This will take some times to finish, and as long as your environment is correctly configured, you will finally get a JSON-format configuration which represents the best kernel version Antares found, then you can do 2 things:

1) Re-evalutation on the case Antares found using `CONFIG` variable:
```sh
    CONFIG='{"axis_0": [-1, 16, 64, 1], "reorder": [0]}' COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' BACKEND=c-cuda make
```

2) If you want to save the result so that frontends like Tensorflow/NNfusion can utilize the optimized kernel, you need to append `COMMIT=1` for your case, like:
```sh
    COMMIT=1 CONFIG='{"axis_0": [-1, 16, 64, 1], "reorder": [0]}' COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' BACKEND=c-cuda make
```
If you want to auto commit the result together with tuning procedure, you can just merge `STEP` and `COMMIT` together:
```sh
    COMMIT=1 STEP=1000 BACKEND=c-cuda COMPUTE_V1='- einstein_v2("output0[N, F, HO, WO] +=! input0[N, C, HO * 4 + KH, WO * 4 + KW] * input1[F, C, KH, KW] where HO in 55, WO in 55", input_dict={"input0": {"dtype": "float32", "shape": [64, 3, 227, 227]}, "input1": {"dtype": "float32", "shape": [96, 3, 11, 11]}});' make
```

After you commit the results, the Antares REST Server will detect this record and response this code version to other frameworks once they newly requests the expression case you saved.

## Remote tunning:

For DirectX12 platform, you could use remote mode to tune expressions. Compared to local tunning, there are two extra things to do:

1) Start remote tuning server on the target machine, reference to [HLSL server](platforms/c-hlsl/evaluator/AntaresEvalAgent)

2) Add server host into the tunning expression
```sh
COMMIT=1 AGENT_URL=${Host_IP}:${Host_Port} CONFIG='{"axis_0": [-1, 16, 64, 1], "reorder": [0]}' COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})' BACKEND=c-hlsl make
```

# How to run Antares REST Server for different platforms:
You can add environment variable `HTTP_PORT=<portnum>` to change the listening port, by default, it will be listening on localhost:8880:
```sh
    BACKEND=c-cuda make rest-server
    BACKEND=c-hlsl make rest-server
    ...
```

# How to use custom tuners as searching algorithms:
Custom tuners can be chosen by adding variable `TUNER=..`, and the value can be selected from any filename under folder `tuner/`, e.g.:
```sh
    TUNER=Ansor STEP=100 BACKEND=c-cuda make
```

# About Microsft Open Source
For more information about Microsoft Open Source Policy, please see [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
