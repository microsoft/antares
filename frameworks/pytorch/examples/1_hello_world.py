#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.contrib.antares.custom_op import CustomOp

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': False}

x = torch.ones(128, 1024, **kwargs)
y = torch.ones(1024, 1024, **kwargs)

custom_op = CustomOp(ir='dot_0[N, M] +=! data[N, K] * channels[K, M]', input_orders={'data': x, 'channels': y}).to(device, dtype).tune(step=100, use_cache=True, timeout=600).emit()

result = custom_op(x, y)
print('The result of tensor `%s` is:\n%s' % (custom_op.output_names[0], result))
