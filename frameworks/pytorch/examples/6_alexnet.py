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

def create_param(name, shape):
  return (torch.rand(shape, **kwargs) - 0.5) * 0.001

input_tensor = torch.ones([64, 3, 227, 227], **kwargs)
const_0_ = create_param('const_0_', [11, 11, 3, 64])
const_1_ = create_param('const_1_', [5, 5, 64, 192])
const_2_ = create_param('const_2_', [3, 3, 192, 384])
const_3_ = create_param('const_3_', [3, 3, 384, 256])
const_4_ = create_param('const_4_', [3, 3, 256, 256])
const_5_ = create_param('const_5_', [9216, 4096])
const_6_ = create_param('const_6_', [4096, 4096])
const_7_ = create_param('const_7_', [4096, 1000])

output_logits = CustomOp(ir=f'''
  conv_0[N, F, HO, WO] +=! input_tensor[N, C, HO * 4 + KH, WO * 4 + KW] * const_0_[KH, KW, C, F] where HO in 55, WO in 55;
  mpool_0[N, C, HO, WO] >=! conv_0[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 27, WO in 27, KH in 3, KW in 3;
  conv_1[N, F, HO, WO] +=! mpool_0[N, C, -2 + HO + KH, -2 + WO + KW].when([-2 + HO + KH >= 0, -2 + HO + KH < 27, -2 + WO + KW >= 0, -2 + WO + KW < 27], 0.0) * const_1_[KH, KW, C, F] where HO in 27, WO in 27;
  mpool_1[N, C, HO, WO] >=! conv_1[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 13, WO in 13, KH in 3, KW in 3;
  conv_2[N, F, HO, WO] +=! mpool_1[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_2_[KH, KW, C, F] where HO in 13, WO in 13;
  conv_2_relu[N, F, HO, WO] = conv_2[N, F, HO, WO].call(`max`, [0.0]);
  conv_3[N, F, HO, WO] +=! conv_2_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_3_[KH, KW, C, F] where HO in 13, WO in 13;
  conv_3_relu[N, F, HO, WO] = conv_3[N, F, HO, WO].call(`max`, [0.0]);
  conv_4[N, F, HO, WO] +=! conv_3_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_4_[KH, KW, C, F] where HO in 13, WO in 13;
  mpool_2[N, C, HO, WO] >=! conv_4[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 6, WO in 6, KH in 3, KW in 3;
  reshape_0[N0, N1] = mpool_2[N0, N1 // 36 % 256, N1 // 6 % 6, N1 % 6] where N1 in 9216;
  dense_0[N, M] +=! reshape_0[N, K] * const_5_[K, M];
  dense_0_relu[N, M] = dense_0[N, M].call(`max`, [0.0]);
  dense_1[N, M] +=! dense_0_relu[N, K] * const_6_[K, M];
  dense_1_relu[N, M] = dense_1[N, M].call(`max`, [0.0]);
  dense_2[N, M] +=! dense_1_relu[N, K] * const_7_[K, M];
''', input_orders={
  'input_tensor': input_tensor,
  'const_0_': const_0_,
  'const_1_': const_1_,
  'const_2_': const_2_,
  'const_3_': const_3_,
  'const_4_': const_4_,
  'const_5_': const_5_,
  'const_6_': const_6_,
  'const_7_': const_7_,
}, device=device).emit()

result = output_logits(input_tensor, const_0_, const_1_, const_2_, const_3_, const_4_, const_5_, const_6_, const_7_)
print('The result of tensor `%s` is:\n%s' % (output_logits.output_names[0], result))

