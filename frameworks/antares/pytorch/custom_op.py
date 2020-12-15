# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib
from torch.autograd import Function
from http import client as http_client

import antares_custom_op

def generate_antares_expression(antares_ir, inputs):
  input_dict, kwargs = {}, {}
  for k, v in inputs:
    input_dict[k] = {
    'dtype': str(v.dtype).split('.')[1],
    'shape': list(v.shape)
    }
    kwargs[k] = v

  input_dict = json.dumps(input_dict)
  return '- einstein_v2("%s", input_dict=%s)' % (antares_ir.replace('"', '`'), input_dict)

def fetch_and_compile_antares_kernel(expression, expr_hash, server_addr):
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

  code_name = 'Antares' + expr_hash
  source_path = '/tmp/antares_torch_%s.cc.kernel.cu' % code_name

  # Compile Kernel object
  with open(source_path, 'w') as fp:
    fp.write(source)
  output_names = [x.split('/')[-1].strip() for x in meta_outputs]
  return output_names, (source, source_path, expr_hash, meta_inputs, meta_outputs)

'''
class CustomFunction(Function):
  @staticmethod
  def forward(ctx, inputs, attributes):
    outputs = antares_custom_op.forward(inputs, *attributes)
    return outputs
'''

class CustomOp(torch.nn.Module):
  def __init__(self, server_addr='localhost:8880'):
    super(CustomOp, self).__init__()
    self.server_addr = server_addr
    self.ops = dict()

  def forward(self, antares_ir, values, keys=[]):
    if not keys:
      keys = [f'input{i}' for i in range(len(values))]
    antares_expr = generate_antares_expression(antares_ir, zip(keys, values))

    expr_hash = hashlib.sha256(antares_expr.encode()).hexdigest()
    if expr_hash in self.ops:
      output_names, attributes = self.ops[expr_hash]
    else:
      output_names, attributes = fetch_and_compile_antares_kernel(antares_expr, expr_hash, self.server_addr)
      self.ops[expr_hash] = output_names, attributes

    self._output_names = output_names
    outputs = antares_custom_op.forward(values, *attributes)
    return outputs
