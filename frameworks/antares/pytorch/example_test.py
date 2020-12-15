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

custom_op = CustomOp(os.environ.get('ANTARES_ADDR', 'localhost:8880')).to(device, dtype)

inputs = {'data': x}
outputs = custom_op('reduce_sum_0[N] +=! data[N, M]', values=list(inputs.values()), keys=list(inputs.keys()))
print('The result of tensor `%s` is:\n%s' % (custom_op._output_names[0], outputs))
