#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.contrib.antares.custom_op import CustomOp

device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

x = torch.randn(1024, 512, **kwargs)
y = torch.randn(1024, 512, **kwargs)

custom_op = CustomOp(os.environ.get('ANTARES_ADDR', 'localhost:8880')).to(device, dtype)
outputs = custom_op('output0[N, M] = input0[N, M] * input1[N, M] + 1234', [x, y])
print(outputs)
