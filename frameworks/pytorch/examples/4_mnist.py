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

x = torch.ones([64, 28 * 28], **kwargs)

def create_param(name, shape):
  return (torch.rand(shape, **kwargs) - 0.5) * 0.01

w0 = create_param('dense_w0', [28 * 28, 512])
b0 = create_param('dense_b0', [512])
w1 = create_param('dense_w1', [512, 512])
b1 = create_param('dense_b1', [512])
w2 = create_param('dense_w2', [512, 10])
b2 = create_param('dense_b2', [10])

custom_op = CustomOp(ir='''
  data_0[N, M] +=!  data[N, K] * weight_0[K, M];
  data_0_bias[N, K] = data_0[N, K] + bias_0[K];
  data_1[N, K] =   data_0_bias[N, K].call(`max`, [0.0]);
  data_2[N, M] +=!  data_1[N, K] * weight_1[K, M];
  data_2_bias[N, K] = data_2[N, K] + bias_1[K];
  data_3[N, K] =   data_2_bias[N, K].call(`max`, [0.0]);
  data_4[N, M] +=!  data_3[N, K] * weight_2[K, M];
  data_5[N, K] =   (data_4[N, K] + bias_2[K]);
''', input_orders={'data': x, 'weight_0': w0, 'weight_1': w1, 'weight_2': w2, 'bias_0': b0, 'bias_1': b1, 'bias_2': b2}).to(device).tune(step=100, use_cache=True, timeout=600).emit()

result = custom_op(x, w0, w1, w2, b0, b1, b2)
print('The result of tensor `%s` is:\n%s' % (custom_op.output_names[0], result))

