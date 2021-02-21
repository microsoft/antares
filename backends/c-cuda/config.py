# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return len(subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split())

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, args, body = codeset
  expand_args = ', '.join([f'{x[0]}* __restrict__ {x[1]}' for x in args])

  def get_extent(key, defval=1):
    str_pat = f'// [thread_extent] {key} = '
    idx = body.find(str_pat)
    if idx >= 0:
      return int(body[idx+len(str_pat):body.index('\n', idx)])
    return defval

  launch_bounds = get_extent('threadIdx.x') * get_extent('threadIdx.y') * get_extent('threadIdx.z')

  full_body = f'''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __HALF_MAXMIN_EX__
#define __HALF_MAXMIN_EX__
inline __device__ half max(half x, half y) {{ return x > y ? x : y; }}
inline __device__ half min(half x, half y) {{ return x < y ? x : y; }}
#endif
{kwargs['attrs'].blend}

extern "C" __global__ __launch_bounds__({launch_bounds}) void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body
