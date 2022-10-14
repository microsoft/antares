# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import importlib
import json
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
  ngpus = len(subprocess.getoutput('/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000').split())
  ngpus = ngpus if ngpus > 0 else 1
  return ngpus

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

  if 'AMDGFX' in os.environ:
    amdgfx = os.environ['AMDGFX']
  else:
    amdgfx = kwargs['attrs'].device_props.compute_version.split('.')
    if int(amdgfx[0]) < 10:
      amdgfx = 'gfx%u%02x' % (int(amdgfx[0]), int(amdgfx[1]))
    else:
      amdgfx = 'gfx%u%02u' % (int(amdgfx[0]), int(amdgfx[1]))

  launch_bounds = get_extent('threadIdx.x') * get_extent('threadIdx.y') * get_extent('threadIdx.z')

  full_body = f'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef __ROCM_COMMON_MACRO__
#define __ROCM_COMMON_MACRO__

#define __ITEM_0_OF__(v) (v).x
#define __ITEM_1_OF__(v) (v).y
#define __ITEM_2_OF__(v) (v).z
#define __ITEM_3_OF__(v) (v).w

#define __STORE_ITEM_0__(t, out, ido, in, idi) *(t*)(out + ido) = *(t*)(in + idi)
#define __STORE_ITEM_1__(t, out, ido, in, idi)
#define __STORE_ITEM_2__(t, out, ido, in, idi)
#define __STORE_ITEM_3__(t, out, ido, in, idi)

#define __AMDGFX__ {amdgfx}

__forceinline__ __device__ __half hmax(const __half &a, const __half &b) {{ return a > b ? a : b; }}
__forceinline__ __device__ __half hmin(const __half &a, const __half &b) {{ return a < b ? a : b; }}

#endif
{kwargs['attrs'].blend}

extern "C" __global__ __launch_bounds__({launch_bounds}) void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body
