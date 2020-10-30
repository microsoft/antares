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
    arg_bufs = AntaresGlobal.current_arg_bufs

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

    for arg in delta_args:
      code = code.replace(arg + '[(0)]', arg)
      func_args += '\n int %s; // delta scaler' % arg

    function_name, codelet_name = 'compute_task', 'Vuid_%s' % hashlib.sha1(code.encode()).hexdigest()

    code = '''
#include <poplar/Vertex.hpp>

using namespace poplar;

namespace {
%s
}

class %s: public Vertex {
public:
 bool compute() {
%s
  return true;
 }

%s
};
''' % (kwargs['attrs'].blend, codelet_name, code, func_args)

    # Attach rules of input slices
    from antares.common import local_get_dir_file
    with open(local_get_dir_file('slices.json'), 'r') as fp:
      slices_info = json.load(fp)

    codelet_buf = ['']
    codelet_buf.append('std::stringstream codelet;')
    codelet_buf.append('codelet << R"(%s)";' % code)
    codelet_buf.append('g.addCodelets(codelet);')
    codelet_buf.append('')

    codelet_buf.append('poplar::VertexRef v;')
    codelet_buf.append('auto compset = g.addComputeSet(__func__);')
    codelet_buf.append('prog.add(poplar::program::Execute(compset));')
    codelet_buf.append('')

    global_result_shape = None
    output_props = arg_bufs['_out'][0]
    ax_names = [x['name'] for x in slices_info['data_axes']]

    for rank, (axis, tensor) in enumerate(slices_info['slices']):
      codelet_buf.append('v = g.addVertex(compset, "%s");' % codelet_name)
      codelet_buf.append('if (g.getTarget().getTargetType() == poplar::TargetType::IPU_MODEL) g.setCycleEstimate(v, 10);')

      for ax in ax_names:
        codelet_buf.append('g.setInitialValue(v["_%s"], %d);' % (ax, axis[ax][0]))

      for k in tensor:
        ls, rs = [], []
        for l, r in tensor[k]:
          ls.append(l)
          rs.append(r + 1)
        codelet_buf.append('g.connect(v["%s"], i.find("%s")->second.slice({%s}, {%s}).flatten());' % (k, k, str(ls)[1:-1], str(rs)[1:-1]))
        stride = [1] * len(ls)
        for i in reversed(range(len(stride) - 1)):
          stride[i] = stride[i + 1] * (rs[i + 1] - ls[i + 1])
        delta_val = int(np.dot(ls, stride))
        codelet_buf.append('g.setInitialValue(v["_%s"], %d);' % (k, delta_val))
      ls, rs = [], []
      for ax in ax_names:
        l, r = axis[ax]
        ls.append(l)
        rs.append(r + 1)
      global_result_shape = rs
      output_slice = 'result.slice({%s}, {%s}).flatten()' % (str(ls)[1:-1], str(rs)[1:-1])
      codelet_buf.append('g.connect(v["%s"], %s);' % (output_props['name'], output_slice))
      codelet_buf.append('g.setTileMapping(%s, %d);' % (output_slice, rank % 1216))
      codelet_buf.append('g.setTileMapping(v, %d);' % (rank % 1216))
      codelet_buf.append('')

    codelet_buf.insert(1, 'poplar::Tensor result = g.addVariable(poplar::%s, poplar::ArrayRef<std::size_t>({%s}), "%s");' % (_native_dtype(output_props['dtype']).upper(), str(global_result_shape)[1:-1], output_props['name']))
    codelet_buf.append('return std::move(result);')

    codelet_buf = '\n  '.join(codelet_buf)
    code = 'poplar::Tensor %s(poplar::Graph &g, poplar::program::Sequence &prog, const std::unordered_map<std::string, poplar::Tensor> &i) {%s\n}' % (function_name, codelet_buf)
    return code
