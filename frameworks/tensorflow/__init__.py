# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
try:
  from tensorflow.contrib.util import loader
except:
  loader = tf

from tensorflow.python.platform import resource_loader

from http import client as http_client
import json, os, hashlib, shutil, time, subprocess

if not tf.test.is_built_with_gpu_support():
  if os.system('which dpcpp >/dev/null') == 0:
    backend = 'c-sycl_intel'
  else:
    backend = 'c-mcpu_avx512' if os.system("grep -r '\\bavx512' /proc/cpuinfo >/dev/null") == 0 else 'c-mcpu'
else:
  is_cuda = tf.test.is_built_with_cuda()
  backend = 'c-cuda' if is_cuda else 'c-rocm'
print(f'[Info] \033[92mInitialize Antares for backend = {backend}\033[0m')

def get_antares_cmd(expression, step=0):
  assert 0 == os.system('which antares >/dev/null 2>&1'), "`antares` command is not found in PATH, have you completed installing antares from pip?"
  commit = 'COMMIT=force' if step > 0 else ''
  return f"BACKEND={backend} STEP={step} {commit} COMPUTE_V1='{expression}' antares"

def get_tensorflow_antares_component(tf_module_path, op_name, compiler):
  dist_path = tf.sysconfig.get_include() + '/..'
  abi_flag = tf.sysconfig.CXX11_ABI_FLAG
  if os.path.exists(f'{tf_module_path}.so.{compiler}'):
    return f'{tf_module_path}.so.{compiler}'

  for flag in tf.sysconfig.get_link_flags():
    if flag.startswith('-l:libtensorflow_framework.so'):
      libtf_so_name = flag[3:].strip()
      break
  if tf.test.is_built_with_rocm():
    with_cuda = "-DANTARES_ROCM -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64"
    if compiler == 'mpicc':
      with_cuda += ' -lmpi_cxx -lrccl'
  elif tf.test.is_built_with_cuda():
    with_cuda = "-DANTARES_CUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcuda"
    if compiler == 'mpicc':
      with_cuda += ' -lmpi_cxx -lnccl'
  elif backend in ('c-sycl_intel',):
    with_cuda = f'-DANTARES_SYCL -D__BACKEND__=\\"{backend}\\" -ldl -lpthread -Wno-string-compare -Wno-unused-value'
    if compiler == 'mpicc':
      with_cuda += ' -lmpicxx'
    compiler = 'dpcpp'
  else:
    with_cuda = f'-DANTARES_MCPU -D__BACKEND__=\\"{backend}\\" -Wno-string-compare -Wno-unused-value'
    if compiler == 'mpicc':
      with_cuda += ' -lmpi_cxx'

  self_pid = os.getpid()
  # Compile TF library
  cmd = f'''{compiler} -pthread -DNDEBUG -g -fwrapv -shared -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC \
    {tf_module_path} -o {tf_module_path}.so.{self_pid} -std=c++11 -fPIC -O2 -DOP_NAME='"{op_name}"' \
    -I{dist_path}/include -L{dist_path}/ -l:{libtf_so_name} -I/usr/local {with_cuda} \
    -I {os.path.dirname(__file__)} \
    -pthread -Wl,-rpath -Wl,--enable-new-dtags -D_GLIBCXX_USE_CXX11_ABI={abi_flag}'''
  if os.system(cmd) != 0:
    raise Exception("Failed to compile the tensorflow plugins: %s" % cmd)

  os.system(f'mv {tf_module_path}.so.{self_pid} {tf_module_path}.so.{compiler} >/dev/null 2>&1 || true')
  return f'{tf_module_path}.so.{compiler}'

__ops_name__ = __loader__.name.split('.')[-1]


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

  ir = ir.replace('"', '`').replace('\n', ' ').strip()
  input_dict = json.dumps(input_dict)
  extra_outputs = ', '.join(['"%s"' % x for x in extra_outputs])
  expression = f'- einstein_v2(input_dict={input_dict}, extra_outputs=[{extra_outputs}], exprss="{ir}")'
  print('+ [Antares Op]', expression)

  def request_code():
    source = subprocess.getoutput(get_antares_cmd(expression))
    try:
      source = source[source.index('// GLOBALS: '):source.rindex('// --------------')]
    except:
      raise Exception(f'[Error] Failed to request code from Antares:\n\n{source}\n')
    return source

  def tune(step=100, use_cache=False, timeout=-1):
    if use_cache and request_code().find('// Saved Perf =') >= 0 or step <= 0:
      return request_code
    cmd = get_antares_cmd(expression, step=step)
    print(f'[Exec] \033[92m{cmd}\033[0m')
    os.system(cmd)
    return request_code

  def emit():
    source = request_code()
    meta_bgn = source.index('// GLOBALS: ') + len('// GLOBALS: ')
    meta_pos = source.index(' -> ', meta_bgn)
    meta_end = source.index('\n', meta_pos)
    meta_inputs = source[meta_bgn:meta_pos - 1].split('], ')
    meta_outputs = source[meta_pos + len(' -> '):meta_end - 1].split('], ')
    kwargs['source'] = source
    kwargs['antares_ir'] = ir

    def parse_tensor(encoded_tensor):
      name, parts = encoded_tensor.split(':')
      dtype, shapes = parts.split('[')
      return name, dtype, [int(x) for x in shapes.split(', ')]

    code_name = 'Antares' + hashlib.sha256(expression.encode()).hexdigest()
    tf_module_path = f'/tmp/antares_tf_{backend}_{code_name}.cc'

    shutil.copyfile(resource_loader.get_path_to_datafile('main_ops.cc.in'), tf_module_path)
    with open(tf_module_path, 'a') as fp:
      fp.write('REGISTER_OP(OP_NAME)')
      for i in range(len(meta_inputs)):
        name, dtype, shape = parse_tensor(meta_inputs[i])
        fp.write(f'\n  .Input("{name}: {dtype}") // {shape}')
      for i in range(len(meta_outputs)):
        name, dtype, shape = parse_tensor(meta_outputs[i])
        fp.write(f'\n  .Output("{name}: {dtype}") // {shape}')
      fp.write('\n  .Attr("source: string").Attr("antares_ir: string").Attr("tf_module_path: string").Attr("meta_inputs: list(string)").Attr("meta_outputs: list(string)").SetIsStateful()')
      fp.write('\n  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {')
      for i in range(len(meta_outputs)):
        name, dtype, shape = parse_tensor(meta_outputs[i])
        fp.write(f'\n    c->set_output({i}, c->MakeShape({{ {str(shape)[1:-1]} }}));')
      fp.write('\n    return ::tensorflow::Status::OK();\n  });')

    libops_path = get_tensorflow_antares_component(tf_module_path, code_name, 'gcc')
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

    output_names = [parse_tensor(x)[0] for x in meta_outputs]
    if len(output_names) == 1:
      result = tf.identity(result, name=output_names[0])
    else:
      result = list(result)
      for i in range(len(result)):
        result[i] = tf.identity(result[i], name=output_names[i])
      result = tuple(result)
    return result

  request_code.tune = tune
  request_code.emit = emit
  return request_code

communicate_library = None

def init_library():
  global communicate_library
  if communicate_library is None:
    libcommunicate_path = get_tensorflow_antares_component(os.path.dirname(__file__) + '/communicate_ops.cc', 'AntaresCommunicate', 'mpicc')
    communicate_library = loader.load_op_library(libcommunicate_path)
  return communicate_library

def init_communicate_config(expect_nodes=None):
  communicate_library = init_library()
  if not hasattr(communicate_library, 'comm_config'):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    rank, size, local_rank = comm.Get_rank(), comm.Get_size(), local.Get_rank()
    MPI.COMM_WORLD.Barrier()
    communicate_library.comm_config = rank, size, local_rank
  if expect_nodes is not None and expect_nodes != communicate_library.comm_config[1]:
    raise Exception(f"The program is designed to use {expect_nodes} nodes, while the environment only detects {communicate_library.comm_config[1]} nodes.")
  return communicate_library.comm_config

def metric(data):
  communicate_library = init_library()
  results = communicate_library.metric(data)
  return results

def synchronize(data):
  communicate_library = init_library()
  results = communicate_library.synchronize(data)
  return results

def communicate(comm_type, data, names=[]):
  rank, size, local_rank = init_communicate_config()
  out = communicate_library.collective(data, op_type=comm_type)
  if comm_type.startswith('all_reduce:'):
    for i in range(len(data)):
      out[i] = tf.reshape(out[i], data[i].shape)
  elif comm_type.startswith('reduce_scatter:'):
    for i in range(len(data)):
      _shape = [int(x) for x in data[i].shape]
      for k in range(len(_shape)):
        if _shape[k] % size == 0:
          _shape[k] //= size
          break
        elif _shape[k] > 1:
          raise f"Tensor of shape {_shape} cannot be performed by reduce_scatter which is divided into {size} pieces."
      out[i] = tf.reshape(out[i], _shape)
  elif comm_type.startswith('all_gather:'):
    dim = int(comm_type[comm_type.index(':') + 1:])
    for i in range(len(data)):
      _shape = [int(x) for x in data[i].shape]
      for k in range(min(len(_shape), dim)):
        assert _shape[k] == 1, f"Tensor of shape {_shape} cannot be performed by all_gather which is aggregated from {size} pieces."
      _shape[dim] *= size
      out[i] = tf.reshape(out[i], _shape)
  else:
    raise Exception(f"Unrecognized communication type: {comm_type}")

  if names:
    out = list(out)
    for i in range(len(names)):
      out[i] = tf.identity(out[i], name=names[i])
    out = tuple(out)
  return out

def communicate_ex(comm_type, data, names=[]):
  if comm_type.startswith('fwd_all_reduce:'):
    @tf.custom_gradient
    def compute(t):
      [t] = communicate(comm_type[4:], [t])
      def grad(dy):
        return dy
      return t, grad
    return compute(data)
  if comm_type.startswith('bwd_all_reduce:'):
    @tf.custom_gradient
    def compute(t):
      def grad(dy):
        [dy] = communicate(comm_type[4:], [dy])
        return dy
      return t, grad
    return compute(data)
  if comm_type.startswith('fwd_reduce_scatter:'):
    @tf.custom_gradient
    def compute(t):
      [t2] = communicate(comm_type[4:], [t])
      for dim in range(len(t.shape)):
        if t.shape[dim] != t2.shape[dim]:
          break
      def grad(dy):
        [dy] = communicate(f'all_gather:{dim}', [dy])
        return dy
      return t2, grad
    return compute(data)
  if comm_type.startswith('fwd_all_gather:'):
    @tf.custom_gradient
    def compute(t):
      [t2] = communicate(comm_type[4:], [t])
      def grad(dy):
        [dy] = communicate(f'reduce_scatter:+', [dy])
        return dy
      return t2, grad
    return compute(data)

  raise Exception(f"Unrecognized communication type: {comm_type}")
