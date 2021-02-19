# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

from antares.common import type_to_c as _native_dtype, AntaresGlobal

def get_execution_parallism():
    return 1

def do_native_translation(code, **kwargs):
    arg_bufs = AntaresGlobal.local_arg_pros

    idx = code.index('(', code.index('extern "C" __global__ ')) + 1
    tail = code.index(') {\n', idx)

    args = []
    for buf in arg_bufs['_in']:
      args.append((_native_dtype(buf['dtype']), buf['name']))
    for buf in arg_bufs['_out']:
      args.append((_native_dtype(buf['dtype']), buf['name']))

    code = 'extern "C" void kernel_main(%s) {\n  // [thread_compute]\n' % ', '.join([t + '* ' + v for t, v in args]) + code[tail + len(") {\n"):]
    code = code.replace('threadIdx.x', '__rank__').replace(' __global__ ', ' ').replace(' __restrict__ ', ' ')
    code = '#include <math.h>\n#include <algorithm>\nusing namespace std;\n\n' + kwargs['attrs'].blend + '\n' + code
    return code
