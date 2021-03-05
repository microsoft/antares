# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os


def get_execution_parallism():
  ngpus = len(subprocess.getoutput('/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000').split())
  ngpus = ngpus if ngpus > 0 else 1
  return ngpus

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'{x[0]}* __restrict__ {x[1]}' for x in in_args + out_args])

  def get_extent(key, defval=1):
    str_pat = f'// [thread_extent] {key} = '
    idx = body.find(str_pat)
    if idx >= 0:
      return int(body[idx+len(str_pat):body.index('\n', idx)])
    return defval

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

#endif
{kwargs['attrs'].blend}

extern "C" __global__ __launch_bounds__({launch_bounds}) void {kernel_name}({expand_args}) {{
  {body}
}}
'''
  return full_body
