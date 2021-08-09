# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib, time, subprocess
from torch.autograd import Function
from http import client as http_client

import antares_custom_op

if not torch.cuda.is_available():
  backend = 'c-mcpu_avx512' if os.system("grep -r '\\bavx512' /proc/cpuinfo >/dev/null") == 0 else 'c-mcpu'
else:
  from torch.utils.cpp_extension import IS_HIP_EXTENSION
  is_cuda = not IS_HIP_EXTENSION
  backend = 'c-cuda' if is_cuda else 'c-rocm'
print(f'[Info] \033[92mInitialize Antares for backend = {backend}\033[0m')

def generate_antares_expression(ir, feed_dict, extra_outputs):
  input_dict, kwargs = {}, {}
  for k in feed_dict:
    v = feed_dict[k]
    input_dict[k] = {
      'dtype': str(v.dtype).split('.')[1],
      'shape': list(v.shape)
    }
    kwargs[k] = v

  ir = ir.replace('"', '`').replace('\n', ' ').strip()
  input_dict = json.dumps(input_dict)
  extra_outputs = ', '.join(['"%s"' % x for x in extra_outputs])
  expression = f'- einstein_v2(input_dict={input_dict}, extra_outputs=[{extra_outputs}], exprss="{ir}")'
  return expression

def get_antares_cmd(expression, step=0):
  antares_local_path = os.environ.get('ANTARES_ROOT')
  assert antares_local_path, "User environment `ANTARES_ROOT` for antares directory is not set, please set it by: export ANTARES_ROOT=<root-path-of-antares>"
  commit = 'COMMIT=force' if step > 0 else ''
  return f"cd '{antares_local_path}' && BACKEND={backend} STEP={step} {commit} COMPUTE_V1='{expression}' make"

class CustomOp(torch.nn.Module):
  __custom_op_dict__ = dict()

  def __init__(self, ir, input_orders, extra_outputs=[]):
    super(CustomOp, self).__init__()
    ir = ir.replace('"', '`').replace('\n', ' ').strip()
    self.expr = generate_antares_expression(ir, input_orders, extra_outputs)
    self.input_orders = sorted([(k, i, input_orders[k].shape, input_orders[k].dtype) for i, k in enumerate(input_orders)], key=lambda x: x[0])
    if not hasattr(CustomOp, '__CUSTOM_KEY__'):
      CustomOp.__CUSTOM_KEY__ = 0
    self.custom_key = CustomOp.__CUSTOM_KEY__
    CustomOp.__CUSTOM_KEY__ += 1

  def request_code(self):
    expression = self.expr
    source = subprocess.getoutput(get_antares_cmd(expression))
    try:
      source = source[source.index('// GLOBALS: '):source.rindex('// --------------')]
    except:
      raise Exception(f'[Error] Failed to request code from Antares:\n\n{source}\n')
    return source

  def fetch_and_compile_antares_kernel(self, expr_hash):
    expression = self.expr
    print('+ [Antares Op]', expression)

    source = self.request_code()
    try:
      meta_bgn = source.index('// GLOBALS: ') + len('// GLOBALS: ')
    except:
      raise Exception("Illegal syntax for Antares expression: %s" % expression)
    meta_pos = source.index(' -> ', meta_bgn)
    meta_end = source.index('\n', meta_pos)
    meta_inputs = source[meta_bgn:meta_pos - 1].split('], ')
    meta_outputs = source[meta_pos + len(' -> '):meta_end - 1].split('], ')

    code_name = 'Antares' + expr_hash
    source_path = f'/tmp/antares_torch_{backend}_{code_name}.cc.kernel.cu'

    # Compile Kernel object
    with open(source_path, 'w') as fp:
      fp.write(source)

    def parse_tensor(encoded_tensor):
      name, parts = encoded_tensor.split(':')
      dtype, shapes = parts.split('[')
      return name, dtype, [int(x) for x in shapes.split(', ')]

    output_names = [parse_tensor(x)[0] for x in meta_outputs]
    return output_names, source

  def tune(self, step=100, use_cache=False, timeout=-1):
    if use_cache and self.request_code().find('// Saved Perf =') >= 0 or step <= 0:
      return self
    expression = self.expr
    cmd = get_antares_cmd(expression, step=step)
    print(f'[Exec] \033[92m{cmd}\033[0m')
    os.system(cmd)
    return self

  def emit(self):
    expr_hash = hashlib.sha256(self.expr.encode()).hexdigest()
    __custom_op_dict__ = CustomOp.__custom_op_dict__
    if expr_hash in __custom_op_dict__:
      output_names, kernel_sources = __custom_op_dict__[expr_hash]
    else:
      output_names, kernel_sources = self.fetch_and_compile_antares_kernel(expr_hash)
      __custom_op_dict__[expr_hash] = output_names, kernel_sources
    self.kernel_sources = kernel_sources
    self.output_names = output_names

    antares_custom_op.forward([], self.custom_key, self.kernel_sources)
    return self

  def forward(self, *inputs):
    ordered_inputs = []
    for i in range(len(inputs)):
      inp = inputs[self.input_orders[i][1]]
      if self.input_orders[i][3] != inp.dtype or self.input_orders[i][2] != inp.shape:
        raise Exception(f"The order of planned inputs ({str(self.input_orders[i][3])}{list(self.input_orders[i][2])}) and given inputs ({str(inp.dtype)}{list(inp.shape)}) doesn't match.")
      ordered_inputs.append(inp)

    outputs = antares_custom_op.forward(ordered_inputs, self.custom_key, '')
    outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return outputs
