# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'__global {x[0]}* {x[1]}' for x in in_args + out_args])

  body = body.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)').replace('__shared__', '__local')
  parsed_lines, body = [], body.split('\n')
  for line in body:
    parts = line.split(' = ')
    if len(parts) == 2:
      parts[0] = parts[0].replace('(float4*)', '(__local float4*)').replace('(float2*)', '(__local float2*)')
      parts[1] = parts[1].replace('(float4*)', '(__global float4*)').replace('(float2*)', '(__global float2*)')
      line = f'{parts[0]} = {parts[1]}'
    parsed_lines.append(line)
  body = '\n'.join(parsed_lines)
  del parsed_lines

  for i, key in enumerate(['blockIdx.x', 'blockIdx.y', 'blockIdx.z']):
    body = body.replace(key, 'get_group_id(%d)' % i)
  for i, key in enumerate(['threadIdx.x', 'threadIdx.y', 'threadIdx.z']):
    body = body.replace(key, 'get_local_id(%d)' % i)
 
  full_body = f'''{kwargs['attrs'].blend}
#ifndef __OCL_COMMON_MACRO__
#define __OCL_COMMON_MACRO__

#define __ITEM_0_OF__(v) (v).x
#define __ITEM_1_OF__(v) (v).y
#define __ITEM_2_OF__(v) (v).z
#define __ITEM_3_OF__(v) (v).w

#define make_int4(x, y, z, w) ((int4)(x, y, z, w))
#define make_int2(x, y) ((int2)(x, y))

#endif

__kernel void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body
