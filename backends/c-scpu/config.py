# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess

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
  s_in_args = [f'auto* {x[1]} = ({x[0]}* __restrict)__args[{i}];' for i, x in enumerate(in_args)]
  s_out_args = [f'auto* {x[1]} = ({x[0]}*)__args[{i + len(in_args)}];' for i, x in enumerate(out_args)]
  expand_args = ' '.join(s_in_args + s_out_args)
  if 'VAMAP' in os.environ:
    for i, x in enumerate(os.environ['VAMAP'].split(',')):
      dtype, name = 'int', x.split(':')[0]
      if '/_' in name:
        dtype, name = name.split('/')
      expand_args += f'auto {name} = *({dtype}*)&__args[{i + len(in_args) + len(out_args)}];'

  full_body = f'''
#include <math.h>
#include <algorithm>
#include <regex>
#define rsqrt(x)  (1.0f / sqrt(x))
{kwargs['attrs'].blend}

extern "C" void {kernel_name}(const int __rank__, void** __args) {{
  {expand_args}
  using namespace std;

  {body.replace('threadIdx.x', '__rank__')}
}}
'''
  return full_body
