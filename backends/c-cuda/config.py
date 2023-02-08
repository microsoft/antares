# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import re
from antares.common import backend, AntaresGlobal

def to_search_space(ast_seq, input_dict, output_dict):
  from antares.default_codegen import codegen
  codegen(ast_seq, input_dict, output_dict, {}, space_only=True)
  space = AntaresGlobal.auto_config.get_config_space()
  return space

def to_kernel_slices(compute_graph, best_config):
  from antares.default_codegen import codegen
  return codegen(*compute_graph, best_config)

def get_execution_parallism():
  num_gpus = len(subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split())
  num_gpus = num_gpus if num_gpus > 0 else 1
  return num_gpus

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'{x[0]}* __restrict__ {x[1]}' for x in in_args + out_args])
  if 'VAMAP' in os.environ:
    expand_args += ', ' + ', '.join([f'int {x.split(":")[0]}' if '/_' not in x.split(":")[0] else x.split(":")[0].replace('/', ' ') for x in os.environ['VAMAP'].split(',')])

  def get_extent(key, defval=1):
    str_pat = f'// [thread_extent] {key} = '
    idx = body.find(str_pat)
    if idx >= 0:
      return int(body[idx+len(str_pat):body.index('\n', idx)])
    return defval

  launch_bounds = get_extent('threadIdx.x') * get_extent('threadIdx.y') * get_extent('threadIdx.z')

  cuda_linux_half = ''
  if '_win64' not in backend:
    cuda_linux_half += '\n__forceinline__ __device__ __half hmax(const __half &a, const __half &b) { return a > b ? a : b; }'
    cuda_linux_half += '\n__forceinline__ __device__ __half hmin(const __half &a, const __half &b) { return a < b ? a : b; }\n'

  blend = kwargs['attrs'].blend
  if re.search(fr'\bATOMIC_ADD\b', body):
    blend += '#define ATOMIC_ADD(x, y) atomicAdd(&(x), y)\n'
  if re.search(fr'\bATOMIC_MAX\b', body):
    blend += '#define ATOMIC_MAX(x, y) atomicMax(&(x), y)\n'

  full_body = f'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __CUDA_COMMON_MACRO__
#define __CUDA_COMMON_MACRO__

#if (__CUDA_ARCH__ >= 600)
{cuda_linux_half.strip()}
#endif

#endif
{blend}

extern "C" __global__ __launch_bounds__({launch_bounds}) void {kernel_name}({expand_args}) {{
  {body}
}}
'''

  full_body = re.sub(r'\bint64_t\b', 'long long', full_body)
  full_body = re.sub(r'\buint64_t\b', 'unsigned long long', full_body)
  full_body = re.sub(r'\bfp16_', 'h', full_body)
  return full_body
