# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess


def get_execution_parallism():
    return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ' '.join([f'{x[0]}* {x[1]} = ({x[0]}*)__args[{i}];' for i, x in enumerate(in_args + out_args)])

  full_body = f'''
#include <math.h>
#include <algorithm>
{kwargs['attrs'].blend}

extern "C" void {kernel_name}(int __rank__, void** __args) {{
  {expand_args}
  using namespace std;

  {body.replace('threadIdx.x', '__rank__')}
}}
'''
  return full_body
