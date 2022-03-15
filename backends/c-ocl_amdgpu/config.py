# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

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
  return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ', '.join([f'__global {x[0]}* {x[1]}' for x in in_args + out_args])

  body = body.replace('__syncthreads()', 'barrier(CLK_LOCAL_MEM_FENCE)').replace('__shared__', '__local')

  pre_defines, post_defines = [''], []
  for line in body.split('\n'):
    if line.strip().startswith('__local '):
      pre_defines.append('  ' + line.strip())
    else:
      post_defines.append(line)
  pre_defines = '\n'.join(pre_defines)
  body, post_defines = '\n'.join(post_defines), None

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

#define __STORE_ITEM_0__(t, out, ido, in, idi) *(__local t*)(out + ido) = *(__global t*)(in + idi)
#define __STORE_ITEM_1__(t, out, ido, in, idi)
#define __STORE_ITEM_2__(t, out, ido, in, idi)
#define __STORE_ITEM_3__(t, out, ido, in, idi)

#define make_int4(x, y, z, w) ((int4)(x, y, z, w))
#define make_int2(x, y) ((int2)(x, y))

#endif

__kernel void {kernel_name}({expand_args}) {{{pre_defines}
  {body}
}}
'''
  return full_body
