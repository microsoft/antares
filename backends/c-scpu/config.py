# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess


def get_execution_parallism():
    return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  s_in_args = [f'auto * {x[1]} = ({x[0]}* __restrict)__args[{i}];' for i, x in enumerate(in_args)]
  s_out_args = [f'auto * {x[1]} = ({x[0]}*)__args[{i + len(in_args)}];' for i, x in enumerate(out_args)]
  expand_args = ' '.join(s_in_args + s_out_args)

  full_body = f'''
#include <math.h>
#include <algorithm>
#define rsqrt(x)  (1.0f / sqrt(x))
{kwargs['attrs'].blend}

extern "C" void {kernel_name}(const int __rank__, void** __args) {{
  {expand_args}
  using namespace std;

  {body.replace('threadIdx.x', '__rank__')}
}}
'''
  return full_body
