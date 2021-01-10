# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

from http import client as http_client
import json, os, hashlib, shutil, time


def get_tensorflow_antares_component(tf_module_path, op_name, using_mpi=False):
  dist_path = tf.sysconfig.get_include() + '/..'
  abi_flag = tf.sysconfig.CXX11_ABI_FLAG
  compiler = 'mpicc' if using_mpi else 'gcc'
  if os.path.exists(f'{tf_module_path}.so.{compiler}'):
    return f'{tf_module_path}.so.{compiler}'
  if os.system('ldd %s/libtensorflow_framework.so.1 2>/dev/null | grep -e libamdhip64 >/dev/null' % dist_path) == 0:
    with_cuda = "-DGOOGLE_CUDA -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64"
    if using_mpi:
      with_cuda += ' -lmpi_cxx -lrccl'
  else:
    with_cuda = "-DGOOGLE_CUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda"
    if using_mpi:
      with_cuda += ' -lmpi_cxx -lnccl'

  self_pid = os.getpid()
  # Compile TF library
  cmd = f'''{compiler} -pthread -DNDEBUG -g -fwrapv -shared -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC \
    {tf_module_path} -o {tf_module_path}.so.{self_pid} -std=c++11 -fPIC -O2 -DOP_NAME='"{op_name}"' \
    -I{dist_path}/include -L{dist_path}/ -l:libtensorflow_framework.so.1 -I/usr/local {with_cuda} \
    -pthread -Wl,-rpath -Wl,--enable-new-dtags -D_GLIBCXX_USE_CXX11_ABI={abi_flag}'''
  if os.system(cmd) != 0:
    raise Exception("Failed to compile the tensorflow plugins: %s" % cmd)

  os.system(f'mv {tf_module_path}.so.{self_pid} {tf_module_path}.so.{compiler} >/dev/null 2>&1 || true')
  return f'{tf_module_path}.so.{compiler}'

__ops_name__ = __loader__.name.split('.')[-1]
__default_server_addr__ = 'localhost:8880'

def set_default_server_addr(server_addr):
  global __default_server_addr__
  __default_server_addr__ = server_addr
  if server_addr.find(':') < 0:
    __default_server_addr__ += ':8880'

def make_op(ir, feed_dict, extra_outputs=[]):
  input_dict, kwargs = {}, {}
  if isinstance(feed_dict, list):
    feed_dict = dict([(f'input{i}', feed_dict[i]) for i in range(len(feed_dict))])
  for k in feed_dict:
    assert k[0].islower(), "Tensor name in Antares IR must start with lower case letter."
    dtype = str(feed_dict[k].dtype.name)
    input_dict[k] = {
      'dtype': dtype[:-4] if dtype.endswith('_ref') else dtype,
      'shape': [int(x) for x in feed_dict[k].shape]
    }
    kwargs[k] = feed_dict[k]

  input_dict = json.dumps(input_dict)
  expression = '- einstein_v2("%s", input_dict=%s, extra_outputs=%s)' % (ir.replace('"', '`'), input_dict, extra_outputs)
  print('+ [Antares Op]', expression)

  def request_server(tune_step=0):
    h = http_client.HTTPConnection(__default_server_addr__, timeout=10)
    try:
      h.request('GET', '/', headers={'COMPUTE_V1': expression, 'STEP': tune_step})
    except:
      raise Exception("Failed to contact with Antares server: %s (not started?)" % __default_server_addr__)
    res = h.getresponse()
    if res.status != 200:
      raise Exception("Fail to get server response, reason: %s" % res.reason)
    response = res.read().decode()
    return response

  def tune(step=100, use_cache=False, timeout=-1):
    if use_cache and request_server().find('// Saved Perf =') >= 0 or step <= 0:
      return request_server
    request_server(tune_step=step)
    timer, timeout = 1, int(timeout)
    while timeout == -1 or timer < timeout:
      source = request_server() + '\n'
      idx = source.find('// Saved Perf = ')
      if idx >= 0:
        status = source[idx:source.index('\n', idx)]
        print('+ [Antares Op]', f'>> tuning status (time = {timer}/{timeout}): {status}', end='\r')
      if source.find('// Antares Tuning Completed') >= 0:
        break
      if not timeout:
        break
      timer += 1
      time.sleep(1)
    print()
    return request_server

  def emit():
    source = request_server()
    try:
      meta_bgn = source.index('///') + len('///')
    except:
      raise Exception("Illegal syntax for Antares expression: %s" % expression)
    meta_pos = source.index(':', meta_bgn)
    meta_end = source.index('\n', meta_pos)
    meta_inputs = source[meta_bgn:meta_pos].split(',')
    meta_outputs = source[meta_pos + 1:meta_end].split(',')
    kwargs['source'] = source
    kwargs['antares_ir'] = ir

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

    output_names = [x.split('/')[-1].strip() for x in meta_outputs]
    if len(output_names) == 1:
      result = tf.identity(result, name=output_names[0])
    else:
      result = list(result)
      for i in range(len(result)):
        result[i] = tf.identity(result[i], name=output_names[i])
      result = tuple(result)
    return result

  request_server.tune = tune
  request_server.emit = emit
  return request_server

communicate_library = None

def init_library():
  global communicate_library
  if communicate_library is None:
    libcommunicate_path = get_tensorflow_antares_component(os.path.dirname(__file__) + '/communicate_ops.cc', 'AntaresCommunicate', using_mpi=True)
    communicate_library = loader.load_op_library(libcommunicate_path)
  return communicate_library

def init_communicate_config():
  communicate_library = init_library()
  if not hasattr(communicate_library, 'comm_config'):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank, size, local_rank = comm.Get_rank(), comm.Get_size(), local.Get_rank()
    MPI.COMM_WORLD.Barrier()
    communicate_library.comm_config = rank, size, local_rank
  return communicate_library.comm_config

def metric(data):
  communicate_library = init_library()
  results = communicate_library.metric(data)
  return results

def communicate(comm_type, data, axes=[], names=[]):
  rank, size, local_rank = init_communicate_config()

  if comm_type.startswith('all_reduce:'):
    ops = comm_type[comm_type.index(':') + 1:]
    out = communicate_library.nccl2_allreduce(data, reduce_type=ops)
  elif comm_type.startswith('reduce_scatter:'):
    ops = comm_type[comm_type.index(':') + 1:]
    out = communicate_library.nccl2_reducescatter(data, reduce_type=ops, node_size=size)
    for i in range(len(data)):
      _shape = [int(x) for x in data[i].shape]
      assert _shape[axes[i]] % size == 0, f"Tensor of shape %s cannot be evenly divided by {size} at axis %d" % (_shape, axes[i])
      _shape[axes[i]] //= size
      out[i] = tf.reshape(out[i], _shape)
  elif comm_type.startswith('all_gather:'):
    out = communicate_library.nccl2_allgather(data, node_size=size)
    for i in range(len(data)):
      _shape = [int(x) for x in data[i].shape]
      _shape[axes[i]] *= size
      out[i] = tf.reshape(out[i], _shape)
  else:
    raise Exception(f"Unrecognized communication type: {comm_type}")

  if names:
    out = list(out)
    for i in range(len(names)):
      out[i] = tf.identity(out[i], name=names[i])
    out = tuple(out)
  return out
