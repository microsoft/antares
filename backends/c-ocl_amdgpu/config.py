# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'__global {x[0]}* {x[1]}' for x in in_args + out_args])

  body = body.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)').replace('__shared__', '__local')
  for i, key in enumerate(['blockIdx.x', 'blockIdx.y', 'blockIdx.z']):
    body = body.replace(key, 'get_group_id(%d)' % i)
  for i, key in enumerate(['threadIdx.x', 'threadIdx.y', 'threadIdx.z']):
    body = body.replace(key, 'get_local_id(%d)' % i)
 
  full_body = f'''{kwargs['attrs'].blend}
__kernel void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body
