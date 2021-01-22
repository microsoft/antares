# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
  return ['/bin/cp', kernel_src, kernel_out]

def do_native_translation(code, **kwargs):
  code = code.replace('extern "C" __global__ ', '__kernel ').replace(' __restrict__ ', ' ').replace('__shared__', '__local')
  code = code.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)')
  for i, key in enumerate(['blockIdx.x', 'blockIdx.y', 'blockIdx.z']):
    code = code.replace(key, 'get_group_id(%d)' % i)
  for i, key in enumerate(['threadIdx.x', 'threadIdx.y', 'threadIdx.z']):
    code = code.replace(key, 'get_local_id(%d)' % i)
  param_end = code.index(') {\n')
  param_begin = code.rindex('(', 0, param_end)
  params = ['__global ' + x.strip() for x in code[param_begin + 1:param_end].split(',')]
  params = ', '.join(params)
  code = code[:param_begin + 1] + params + code[param_end:]
  return code
