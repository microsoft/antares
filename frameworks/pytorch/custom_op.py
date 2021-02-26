# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib, time
from torch.autograd import Function
from http import client as http_client

import antares_custom_op

__default_server_addr__ = 'localhost:8880'

def generate_antares_expression(antares_ir, feed_dict, extra_outputs):
  input_dict, kwargs = {}, {}
  for k in feed_dict:
    v = feed_dict[k]
    input_dict[k] = {
      'dtype': str(v.dtype).split('.')[1],
      'shape': list(v.shape)
    }
    kwargs[k] = v
  input_dict = json.dumps(input_dict)
  return '- einstein_v2("%s", input_dict=%s, extra_outputs=%s)' % (antares_ir.replace('"', '`'), input_dict, extra_outputs)

def set_default_server_addr(server_addr):
  global __default_server_addr__
  __default_server_addr__ = server_addr
  if server_addr.find(':') < 0:
    __default_server_addr__ += ':8880'

class CustomOp(torch.nn.Module):
  __custom_op_dict__ = dict()

  def __init__(self, ir, feed_dict, extra_outputs=[]):
    super(CustomOp, self).__init__()
    ir = ir.replace('"', '`').replace('\n', ' ').strip()
    self.expr = generate_antares_expression(ir, feed_dict, extra_outputs)
    feed_dict = sorted([(k, feed_dict[k]) for k in feed_dict], key=lambda x: x[0])
    self.values = [v for (k, v) in feed_dict]

    expr_hash = hashlib.sha256(self.expr.encode()).hexdigest()
    __custom_op_dict__ = CustomOp.__custom_op_dict__
    if expr_hash in __custom_op_dict__:
      output_names, attributes = __custom_op_dict__[expr_hash]
    else:
      output_names, attributes = self.fetch_and_compile_antares_kernel(expr_hash)
      __custom_op_dict__[expr_hash] = output_names, attributes
    self.attributes = attributes
    self.output_names = output_names

  def request_server(self, tune_step=0):
    h = http_client.HTTPConnection(__default_server_addr__, timeout=10)
    try:
      h.request('GET', '/', headers={'COMPUTE_V1': self.expr, 'STEP': tune_step})
    except:
      raise Exception("Failed to contact with Antares server: %s (not started?)" % __default_server_addr__)
    res = h.getresponse()
    if res.status != 200:
      raise Exception("Fail to get server response, reason: %s" % res.reason)
    response = res.read().decode()
    if response.startswith('[ERROR]'):
      raise Exception("IR Compilation Failed - %s" % response)
    return response

  def fetch_and_compile_antares_kernel(self, expr_hash):
    expression = self.expr
    print('+ [Antares Op]', expression)

    source = self.request_server()
    try:
      meta_bgn = source.index('// GLOBALS: ') + len('// GLOBALS: ')
    except:
      raise Exception("Illegal syntax for Antares expression: %s" % expression)
    meta_pos = source.index(' -> ', meta_bgn)
    meta_end = source.index('\n', meta_pos)
    meta_inputs = source[meta_bgn:meta_pos - 1].split('], ')
    meta_outputs = source[meta_pos + len(' -> '):meta_end - 1].split('], ')

    code_name = 'Antares' + expr_hash
    source_path = '/tmp/antares_torch_%s.cc.kernel.cu' % code_name

    # Compile Kernel object
    with open(source_path, 'w') as fp:
      fp.write(source)

    def parse_tensor(encoded_tensor):
      name, parts = encoded_tensor.split(':')
      dtype, shapes = parts.split('[')
      return name, dtype, [int(x) for x in shapes.split(', ')]

    output_names = [parse_tensor(x)[0] for x in meta_outputs]
    return output_names, (source, source_path, expr_hash, meta_inputs, meta_outputs)

  def tune(self, step=100, use_cache=False, timeout=-1):
    if use_cache and self.request_server().find('// Saved Perf =') >= 0 or step <= 0:
      return self
    self.request_server(tune_step=step)
    timer, timeout, status = 1, int(timeout), ''
    while timeout == -1 or timer < timeout:
      try:
        source = self.request_server() + '\n'
      except:
        source = ''
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
    return self

  def emit(self):
    return self

  def forward(self):
    outputs = antares_custom_op.forward(self.values, *self.attributes)
    for i in range(len(outputs)):
      outputs[i].id = self.output_names[i]
    outputs = outputs[0] if len(outputs) == 1 else tuple(outputs)
    return outputs
