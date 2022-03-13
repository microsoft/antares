#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from antares_core.frameworks.pytorch.custom_op import CustomOp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

input0 = torch.ones(1024, 512, **kwargs)
input1 = torch.ones(512, 512, **kwargs)

custom_op = CustomOp(ir='temp0[K, N] = input0[N, K] + 100; output0[N, M] +=! temp0[K, N] * input1[K, M] where K in 10', input_orders={'input0': input0, 'input1': input1}, device=device).tune(step=100, use_cache=True, timeout=600).emit()

result = custom_op(input0, input1)
print('The result of tensor `%s` is:\n%s' % (custom_op.output_names[0], result))
