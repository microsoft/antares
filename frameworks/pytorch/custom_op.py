# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib, time, subprocess
from torch.autograd import Function
import importlib

def get_backend(custom_op):
  device_name = str(custom_op.custom_device)
  if 'cpu' in device_name:
    backend = 'c-mcpu_avx512' if os.system("grep -r '\\bavx512' /proc/cpuinfo >/dev/null") == 0 else 'c-mcpu'
  elif 'cuda' in device_name:
    from torch.utils.cpp_extension import IS_HIP_EXTENSION
    backend = 'c-cuda' if not IS_HIP_EXTENSION else 'c-rocm'
  else:
    raise Exception(f'Unrecognized device name of custom op: {backend}')
  return backend

def generate_antares_expression(ir, feed_list, extra_outputs):
  input_dict, kwargs = {}, {}
  for k, i, shape, dtype in feed_list:
    input_dict[k] = {
      'dtype': str(dtype).split('.')[1],
      'shape': list(shape)
    }

  ir = ir.replace('"', '`').replace('\n', ' ').strip()
  input_dict = json.dumps(input_dict)
  extra_outputs = ', '.join(['"%s"' % x for x in extra_outputs])
  expression = f'- einstein_v2(input_dict={input_dict}, extra_outputs=[{extra_outputs}], exprss="{ir}")'
  return expression

def get_antares_cmd(custom_op, expression, step=0):
  assert 0 == os.system('which antares >/dev/null 2>&1'), "`antares` command is not found in PATH, have you completed installing antares from pip?"
  commit = 'COMMIT=force' if step > 0 else ''
  return f"BACKEND={get_backend(custom_op)} STEP={step} {commit} COMPUTE_V1='{expression}' antares"

class CustomOp(torch.nn.Module):

  def __init__(self, ir, input_orders, extra_outputs=[], device=None):
    super(CustomOp, self).__init__()
    ir = ir.replace('"', '`').replace('\n', ' ').strip()
    input_list, index = [], 0
    for k in input_orders:
      if isinstance(input_orders[k], tuple):
        input_list += [(k, index, input_orders[k][2], input_orders[k][1])]
      else:
        input_list += [(k, index, input_orders[k].shape, input_orders[k].dtype)]
      index += 1

    self.input_orders = sorted(input_list, key=lambda x: x[0])
    self.expr = generate_antares_expression(ir, input_list, extra_outputs)
    self.global_sig = '// GLOBALS: '

    self.custom_device = 'cpu' if device is None else device.type
    backend = get_backend(self)
    lib_name = 'antares_custom_torch_v2_%s' % backend.replace('-', '_')
    try:
      self.custom_lib = importlib.import_module(lib_name)
    except:
      print(f'Failed to import {lib_name}.\nPlease install Custom Plugin for backend in advance: BACKEND={backend} antares torch-setup')
      exit(1)

  def request_code(self):
    expression = self.expr
    source = subprocess.getoutput(get_antares_cmd(self, expression))
    try:
      source = source[source.index(self.global_sig):source.rindex('// --------------')]
    except:
      raise Exception(f'[Error] Failed to request code from Antares:\n\n{source}\n')
    return source

  def emit(self):
    expr_hash = hashlib.sha256(self.expr.encode()).hexdigest()
    expression = self.expr
    print(f'+ [AntaresOp:{get_backend(self)}]', expression)

    source = self.request_code()
    try:
      meta_bgn = source.index(self.global_sig) + len(self.global_sig)
    except:
      raise Exception("Illegal syntax for Antares expression: %s" % expression)

    meta_pos = source.index(' -> ', meta_bgn)
    meta_end = source.index('\n', meta_pos)
    meta_inputs = source[meta_bgn:meta_pos - 1].split('], ')
    meta_outputs = source[meta_pos + len(' -> '):meta_end - 1].split('], ')

    code_name = 'Antares' + expr_hash
    source_path = f'/tmp/antares_torch_{get_backend(self)}_{code_name}.cc.kernel.cu'

    # Compile Kernel object
    with open(source_path, 'w') as fp:
      fp.write(source)

    dtype_mapping = {
      'float64': torch.float64,
      'float32': torch.float32,
      'float16': torch.float16,
      'int64': torch.int64,
      'int32': torch.int32,
      'int16': torch.int16,
      'int8': torch.int8,
    }

    def parse_tensor(encoded_tensor):
      name, parts = encoded_tensor.split(':')
      dtype, shapes = parts.split('[')
      return name, dtype_mapping[dtype], [int(x) for x in shapes.split(', ')]

    output_infos = [parse_tensor(x) for x in meta_outputs]

    self.kernel_sources = source
    self.output_infos = output_infos
    self.output_names = [x[0] for x in output_infos]

    self.custom_key = self.custom_lib.inject(self.kernel_sources)
    return self

  def to(self, *args, **kwargs):
    raise Exception('Deprecated Usage: Legacy `CustomOp(..).to(device)` has been simplified to `CustomOp(.., device=device)`')

  def tune(self, step=100, use_cache=False, timeout=-1):
    if use_cache and self.request_code().find('// Saved Perf =') >= 0 or step <= 0:
      return self
    expression = self.expr
    cmd = get_antares_cmd(self, expression, step=step)
    print(f'[Exec] \033[92m{cmd}\033[0m')
    os.system(cmd)
    return self

  def output(self, index=0):
    return self.output_infos[index]

  def forward(self, *inputs):
    ordered_inputs = []
    for i in range(len(inputs)):
      inp = inputs[self.input_orders[i][1]]
      ordered_inputs.append(inp.contiguous().to(self.custom_device))

    outputs = []
    for info in self.output_infos:
      out = torch.empty(info[2], device=self.custom_device, dtype=info[1])
      outputs.append(out)
    self.custom_lib.forward(self.custom_key, ordered_inputs + outputs)
    outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return outputs
