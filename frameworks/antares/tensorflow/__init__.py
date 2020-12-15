# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

from http import client as http_client
import json, os, hashlib, shutil

def get_tensorflow_antares_component(tf_module_path, op_name):
  dist_path = tf.sysconfig.get_include() + '/..'
  abi_flag = tf.sysconfig.CXX11_ABI_FLAG
  if os.system('ldd %s/libtensorflow_framework.so.1 2>/dev/null | grep -e libamdhip64 >/dev/null' % dist_path) == 0:
    with_cuda = "-DGOOGLE_CUDA -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64"
  else:
    with_cuda = "-DGOOGLE_CUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda"

  # Compile TF library
  cmd = '''gcc -pthread -DNDEBUG -g -fwrapv -shared -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC \
    %s \
    -o %s.so -std=c++11 -fPIC -O2 -DOP_NAME='"%s"' \
    -I%s/include -L%s/ -l:libtensorflow_framework.so.1 \
    -I/usr/local %s \
    -pthread -Wl,-rpath -Wl,--enable-new-dtags -D_GLIBCXX_USE_CXX11_ABI=%d''' % (tf_module_path, tf_module_path, op_name, dist_path, dist_path, with_cuda, abi_flag)

  if os.system(cmd) != 0:
    raise Exception("Failed to compile the tensorflow plugins: %s" % cmd)
  return '%s.so' % tf_module_path

__ops_name__ = __loader__.name.split('.')[-1]
__default_server_addr__ = 'localhost:8880'

def make_op(antares_ir, inputs, server_addr=None):
  if server_addr is None:
    server_addr = __default_server_addr__
  input_dict, kwargs = {}, {}
  if isinstance(inputs, list):
    inputs = dict([(f'input{i}', inputs[i]) for i in range(len(inputs))])
  for k in inputs:
    assert k[0].islower(), "Tensor name in Antares IR must start with lower case letter."
    dtype = str(inputs[k].dtype.name)
    input_dict[k] = {
      'dtype': dtype[:-4] if dtype.endswith('_ref') else dtype,
      'shape': [int(x) for x in inputs[k].shape]
    }
    kwargs[k] = inputs[k]

  input_dict = json.dumps(input_dict)
  expression = '- einstein_v2("%s", input_dict=%s)' % (antares_ir.replace('"', '`'), input_dict)
  print('+ [Antares Op]', expression)

  h = http_client.HTTPConnection(server_addr, timeout=10)
  try:
    h.request('GET', '/', headers={'COMPUTE_V1': expression})
  except:
    raise Exception("Failed to contact with Antares server: %s (not started?)" % server_addr)
  res = h.getresponse()
  if res.status != 200:
    raise Exception("Fail to get server response, reason: %s" % res.reason)

  source = res.read().decode()
  try:
    meta_bgn = source.index('///') + len('///')
  except:
    raise Exception("Illegal syntax for Antares expression: %s" % expression)
  meta_pos = source.index(':', meta_bgn)
  meta_end = source.index('\n', meta_pos)
  meta_inputs = source[meta_bgn:meta_pos].split(',')
  meta_outputs = source[meta_pos + 1:meta_end].split(',')
  kwargs['source'] = source
  kwargs['antares_ir'] = antares_ir 

  code_name = 'Antares' + hashlib.sha256(expression.encode()).hexdigest()
  tf_module_path = '/tmp/antares_tf_%s.cc' % code_name

  shutil.copyfile(resource_loader.get_path_to_datafile('main_ops.cc.in'), tf_module_path)
  with open(tf_module_path, 'a') as fp:
    fp.write('REGISTER_OP(OP_NAME)')
    for i in range(len(meta_inputs)):
      shape, dtype, name = meta_inputs[i].split('/')
      fp.write('\n  .Input("%s: %s") // %s' % (name, dtype, shape.replace('-', ', ')))
    for i in range(len(meta_outputs)):
      shape, dtype, name = meta_outputs[i].split('/')
      fp.write('\n  .Output("%s: %s") // %s' % (name, dtype, shape.replace('-', ', ')))
    fp.write('\n  .Attr("source: string").Attr("antares_ir: string").Attr("tf_module_path: string").Attr("meta_inputs: list(string)").Attr("meta_outputs: list(string)").SetIsStateful()')
    fp.write('\n  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {')
    for i in range(len(meta_outputs)):
      fp.write('\n    c->set_output(%d, c->MakeShape({%s}));' % (i, meta_outputs[i].split('/')[0].replace('-', ', ')))
    fp.write('\n    return ::tensorflow::Status::OK();\n  });')

  libops_path = get_tensorflow_antares_component(tf_module_path, code_name)
  library = loader.load_op_library(libops_path)
  antares_func = None
  for attr in dir(library):
    if attr.startswith('antares') and '_eager' not in attr:
      antares_func = getattr(library, attr)
      break
  if not antares_func:
    raise Exception("Invalid antares component is made.")

  kwargs['tf_module_path'] = tf_module_path
  kwargs['meta_inputs'] = meta_inputs
  kwargs['meta_outputs'] = meta_outputs
  result = antares_func(**kwargs)

  result._output_names = [x.split('/')[-1].strip() for x in meta_outputs]
  result._antares_props = {
    'COMPUTE_V1': expression
  }
  return result
