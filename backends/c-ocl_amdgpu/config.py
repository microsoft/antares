# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'__global {x[0]}* {x[1]}' for x in in_args + out_args])

  body = body.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)').replace('__shared__', '__local')
  body = body.replace('(make_int4)', 'make_int4').replace('(make_int2)', 'make_int2')
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
#ifndef __MAKE_DATA_ARRAY__
#define __MAKE_DATA_ARRAY__
#define make_int4(x, y, z, w) ((int4)(x, y, z, w))
#define make_float4(x, y, z, w) ((float4)(x, y, z, w))
#define make_int2(x, y) ((int2)(x, y))
#define make_float2(x, y) ((float2)(x, y))
#endif

__kernel void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body