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

input0 = torch.ones(1024 * 512, **kwargs)
input1 = torch.ones(1024 * 512, **kwargs)

custom_op = CustomOp(ir='output0[N] = input0[N] + input1[N]; output1[N] = input0[N].call(`exp`); output2[N] = input1[N] + output1[N];', extra_outputs=['output0', 'output1', 'output2'], input_orders={'input0': input0, 'input1': input1}, device=device).tune(step=100, use_cache=True, timeout=600).emit()

result = custom_op(input0, input1)
print('The result of tensor `%s, %s` is:\n%s' % (custom_op.output_names[0], custom_op.output_names[1], result))
