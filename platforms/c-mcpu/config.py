# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess

from antares.common import type_to_c as _native_dtype, AntaresGlobal

def get_execution_parallism():
    return 1

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
    return ['/bin/cp', kernel_src, kernel_out]

def allow_concurrent_compile_execution():
    return False

def remove_local_cache(code, arg_bufs):
    result = []
    for line in code.split('\n'):
      if line.endswith('];') and line.find('=') < 0:
        output_buf = arg_bufs['_out'][0]
        print(line.split()[0], output_buf['dtype'])
        if line.split()[0] != _native_dtype(output_buf['dtype']):
          raise Exception("This backend doesn't support injective computation modifying the output type")
        line = '  ' + line.split('[')[0].strip().replace(' ', ' *') + ' = &' + output_buf['name'] + '[0];'
      result.append(line)
    return '\n'.join(result)

def do_native_translation(code, **kwargs):
    arg_bufs = AntaresGlobal.current_arg_bufs

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
    code = remove_local_cache(code, arg_bufs)
    return code
