#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import torch
from torch.contrib.antares.custom_op import CustomOp

device = torch.device("cuda")
dtype = torch.float32
custom_op = CustomOp().to(device, dtype)

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

x = torch.ones(128, 1024, **kwargs)
y = torch.ones(1024, 1024, **kwargs)

result = custom_op(ir='dot_0[N, M] +=! data[N, K] * weight[K, M]', mapped_keys=['data', 'weight'], mapped_values=[x, y])
print('The result of tensor `%s` is:\n%s' % (result.id, result))
