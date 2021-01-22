# What is Antares:
- Antares is an automatic engine for multi-platform kernel generation and optimization (targeting to CUDA/ROCm/CPU/DirectX12/Graphcore).
- Antares simplifies most TVM's low-level features, making it easier to use for DNN developers on Microsoft related platforms.
- Antares follows "_One Language Syntax for All Platforms_" principle to reduce the description complexity on different platforms.

## Documentation for Quick Start
1. [Quick Start to install Antares for DirectX12](backends/c-hlsl/evaluator/AntaresEvalAgent)
2. [Quick Start to write A + B shaders for DirectX12 using CUDA-like interfaces](backends/c-hlsl/evaluator/AntaresHlslLib/examples)
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

# Reference - Recommended Installation Package Choices for Tensorflow 1.x & 2.x (tested in Ubuntu 20.04):
#   Tensorflow-1 for NVIDIA CUDA 10.0: python3 -m pip install --upgrade pip && python3 -m pip install tensorflow-gpu==1.15.4
#   Tensorflow-1 for NVIDIA CUDA 11.0: python3 -m pip install --upgrade pip && python3 -m pip install https://github.com/ghostplant/tensorflow-wheel-collections/releases/download/cuda-11/tensorflow_gpu-1.15.4_cuda11+nv-cp38-cp38-linux_x86_64.whl
#   Tensorflow-2 for NVIDIA CUDA 11.0: python3 -m pip install --upgrade pip && python3 -m pip install tensorflow-gpu==2.4.0
#   Tensorflow-1 for AMD ROCm 4.0:  python3 -m pip install tensorflow-rocm==1.15.9
#   Tensorflow-2 for AMD ROCm 4.0:  python3 -m pip install tensorflow-rocm==2.4.0

# If you need Antares to extend/boost Pytorch operators, please also run:
sudo python3 ./frameworks/antares/pytorch/setup.py

# Reference - Recommended Installation Package Choices for Pytorch (tested in Ubuntu 20.04):
#   Pytorch for NVIDIA CUDA 10.0: python3 -m pip install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
#   Pytorch for NVIDIA CUDA 11.0: python3 -m pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#   Pytorch for AMD ROCm 4.0:  python3 -m pip install --pre torch==1.8.0.dev20210106 -f https://download.pytorch.org/whl/nightly/rocm4.0/torch_nightly.html
```

# Example with Tensorflow-GPU/Pytorch-GPU:

This example shows you an easy way to quickly add custom operators in Tensorflow/Pytorch, but the operator itself is not an optimized version (not tuned).

```sh
# First, launch the antares REST server (a CUDA example)

BACKEND=c-cuda make rest-server
```

- Tensorflow Frontend Only (>= 1.15.x / >= 2.4.x):
```py
# For Tensorflow CUDA frontend, execute the following python script:

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

x = tf.get_variable('x', [128, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)
y = tf.get_variable('y', [1024, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

op = antares.make_op(ir='dot_0[N, M] +=! data[N, K] * weight[K, M]', feed_dict={'data': x, 'weight': y}).tune(step=100, use_cache=True, timeout=600).emit()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print('The result of tensor `%s` is:\n%s' % (op, sess.run(op)))

```

- Pytorch Frontend Only:
```py
# For Pytorch frontend, execute the following python script:
import torch
from torch.contrib.antares.custom_op import CustomOp

device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

x = torch.ones(128, 1024, **kwargs)
y = torch.ones(1024, 1024, **kwargs)

custom_op = CustomOp(ir='dot_0[N, M] +=! data[N, K] * weight[K, M]', feed_dict={'data': x, 'weight': y}).to(device, dtype).tune(step=100, use_cache=True, timeout=600).emit()

result = custom_op()
print('The result of tensor `%s` is:\n%s' % (result.id, result))
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

## Tunning DirectX12 Compute Shader:

For DirectX12 platform, you could use "Win10 as server + Linux/WSL as client" mode to tune expressions. Please refer documentation [here](backends/c-hlsl/evaluator/AntaresEvalAgent).

# How to run Antares REST Server for different backends:
You can add environment variable `HTTP_PORT=<portnum>` to change the listening port, by default, it will be listening on localhost:8880:
```sh
    HTTP_PORT=8880 BACKEND=c-cuda make rest-server
    HTTP_PORT=8881 BACKEND=c-hlsl make rest-server
    ...
```

# How to use custom tuners as searching algorithms:
Custom tuners can be chosen by adding variable `TUNER=..`, and the value can be selected from any filename under folder `tuner/`, e.g.:
```sh
    TUNER=Ansor STEP=100 BACKEND=c-cuda make
```

# About Microsft Open Source
For more information about Microsoft Open Source Policy, please see [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
