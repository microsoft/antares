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

x = torch.ones(2, 32, **kwargs)
y = torch.ones(32, 32, **kwargs)

custom_op = CustomOp(ir='dot_0[N, M] +=! data[N, K] * channels[K, M]', input_orders={'data': x, 'channels': y}, device=device).tune(step=100, use_cache=True, timeout=600).emit()

torch.manual_seed(0)
for i in range(4):
    A = torch.randn(2, 32, **kwargs)
    B = torch.randn(32, 32, **kwargs)
    C = custom_op(A, B)
    print(f'LOOP-{i}: {C.view(-1)}')

