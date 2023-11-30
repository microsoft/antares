#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import autort

def init_op():
  autort.export(ir="space[data[N] % 2, N] = 1", \
    inputs=["data=int32[N:1000]", "space=int32[2, N:1000]"], config="~N~:[1,256,1]", name="my_classify_i32")

  autort.export(ir="output[ids[N] - 1] = (val[N % val.size(0)]).when(ids[N] > 0)", \
    inputs=["ids=int32[N:1000]", "val=int32[M:2000]", "output=int32[M:2000]"], config="~N~:[1,256,1]", name="my_bucket_i32")

def main(array):
  init_op()

  device = autort.device()
  x = torch.tensor(array, dtype=torch.int32, device=device)
  y = torch.zeros([2, x.size(0)], dtype=torch.int32, device=device)
  autort.ops.my_classify_i32(x, y)

  ids = torch.cumsum(y.view(-1), 0, dtype=y.dtype) * y.view(-1)

  output = torch.zeros_like(x)
  autort.ops.my_bucket_i32(ids, x, output)

  print('\nInput :', x)
  print('  (is_even)', x % 2 == 0)
  print('\nOutput:', output)
  print('  (is_even)', output % 2 == 0)


if __name__ == '__main__':
  main([101, 102, 208, 99, 1, 127, 62, 8, 336, 336])
