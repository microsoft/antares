# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import re

def to_search_space(ast_seq, input_dict, output_dict):
  from antares.default_codegen import codegen
  from antares.common import AntaresGlobal
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
    expand_args += ', ' + ', '.join([f'int {x.split(":")[0]}' for x in os.environ['VAMAP'].split(',')])

  def get_extent(key, defval=1):
    str_pat = f'// [thread_extent] {key} = '
    idx = body.find(str_pat)
    if idx >= 0:
      return int(body[idx+len(str_pat):body.index('\n', idx)])
    return defval

  launch_bounds = get_extent('threadIdx.x') * get_extent('threadIdx.y') * get_extent('threadIdx.z')

  full_body = f'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#ifndef __CUDA_COMMON_MACRO__
#define __CUDA_COMMON_MACRO__

#define __ITEM_0_OF__(v) (v).x
#define __ITEM_1_OF__(v) (v).y
#define __ITEM_2_OF__(v) (v).z
#define __ITEM_3_OF__(v) (v).w

#define __STORE_ITEM_0__(t, out, ido, in, idi) *(t*)(out + ido) = *(t*)(in + idi)
#define __STORE_ITEM_1__(t, out, ido, in, idi)
#define __STORE_ITEM_2__(t, out, ido, in, idi)
#define __STORE_ITEM_3__(t, out, ido, in, idi)

#define MAKE_VEC4_OP(type) \\
  __forceinline__ __device__ type operator+(const type &l, const type &r) {{ return make_##type(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w); }} \\
  __forceinline__ __device__ type operator-(const type &l, const type &r) {{ return make_##type(l.x - r.x, l.y - r.y, l.z - r.z, l.w - r.w); }} \\
  __forceinline__ __device__ type operator*(const type &l, const type &r) {{ return make_##type(l.x * r.x, l.y * r.y, l.z * r.z, l.w * r.w); }} \\
  __forceinline__ __device__ type operator/(const type &l, const type &r) {{ return make_##type(l.x / r.x, l.y / r.y, l.z / r.z, l.w / r.w); }} \\
  __forceinline__ __device__ type operator%(const type &l, const type &r) {{ return make_##type(l.x % r.x, l.y % r.y, l.z % r.z, l.w % r.w); }}
#define MAKE_VEC2_OP(type) \\
  __forceinline__ __device__ type operator+(const type &l, const type &r) {{ return make_##type(l.x + r.x, l.y + r.y); }} \\
  __forceinline__ __device__ type operator-(const type &l, const type &r) {{ return make_##type(l.x - r.x, l.y - r.y); }} \\
  __forceinline__ __device__ type operator*(const type &l, const type &r) {{ return make_##type(l.x * r.x, l.y * r.y); }} \\
  __forceinline__ __device__ type operator/(const type &l, const type &r) {{ return make_##type(l.x / r.x, l.y / r.y); }} \\
  __forceinline__ __device__ type operator%(const type &l, const type &r) {{ return make_##type(l.x % r.x, l.y % r.y); }}

MAKE_VEC4_OP(int4)
MAKE_VEC2_OP(int2)

__forceinline__ __device__ __half max(const __half a, const __half b) {{ return a > b ? a : b; }}
__forceinline__ __device__ __half min(const __half a, const __half b) {{ return a < b ? a : b; }}

#endif
{kwargs['attrs'].blend}

extern "C" __global__ __launch_bounds__({launch_bounds}) void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  if kwargs['attrs'].backend.endswith('_win64'):
    full_body = re.sub(r'\bint64_t\b', '__int64', full_body)
  return full_body
