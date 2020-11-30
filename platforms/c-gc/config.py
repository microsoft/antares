# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import hashlib
import numpy as np

from antares.common import type_to_c as _native_dtype, AntaresGlobal

def get_execution_parallism():
    return 1

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
    return ['/bin/cp', kernel_src, kernel_out]

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
    arg_bufs = AntaresGlobal.local_arg_pros

    if 'einstein_v2' not in kwargs['attrs'].ir:
      raise Exception("Program for graphcore must be based on Antares IR")

    code = code[code.index(') {\n') + len(') {\n'):code.rindex('}\n')]
    code = remove_local_cache(code, arg_bufs)

    func_args, delta_args = '', []
    for buf in arg_bufs['_in']:
      if buf['name'].startswith('_'):
        delta_args.append(buf['name'])
        continue
      func_args += ' Input<Vector<%s>> %s; // local size: %s\n' % (_native_dtype(buf['dtype']), buf['name'], buf['shape'])
    for buf in arg_bufs['_out']:
      func_args += ' Output<Vector<%s>> %s; // local size: %s\n' % (_native_dtype(buf['dtype']), buf['name'], buf['shape'])

    function_name, codelet_name = 'compute_task', 'Vuid_%s' % hashlib.sha1(code.encode()).hexdigest()
    blend_code = kwargs['attrs'].blend.strip()
    blend_code = 'namespace {\n%s\n}\n\n' if blend_code else ''

    from antares.common import local_get_dir_file
    try:
      with open(local_get_dir_file('range_book.json'), 'r') as fp:
        range_book = json.load(fp)
    except FileNotFoundError:
      raise Exception("TODO: Graphcore code generation is not implemented in new emit_tvm_ir_v2()")

    props = []
    for k in range_book['book']:
      arr2d = range_book['book'][k]
      arr2d = [str(x)[1:-1].replace(', ', ',') for x in arr2d]
      arr2d = '/'.join(arr2d)
      props.append(k + '/' + arr2d)
    props = ';'.join(props)

    code = '''
// Antares Property: %s

#include <poplar/Vertex.hpp>

using namespace poplar;

%s
class %s: public Vertex {
public:
 bool compute() {
%s
  return true;
 }

%s
};
''' % (props, blend_code, codelet_name, code, func_args)
    return code

