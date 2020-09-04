# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib
from torch.autograd import Function
from http import client as http_client

import antares_custom_op

def generate_antares_expression(antares_ir, inputs):
  input_dict, kwargs = {}, {}
  for i in range(len(inputs)):
    input_dict['input%d' % i] = {
    'dtype': str(inputs[i].dtype).split('.')[1],
    'shape': list(inputs[i].shape)
    }
    kwargs['input%d' % i] = inputs[i]

  input_dict = json.dumps(input_dict)
  return '- einstein_v2("%s", input_dict=%s)' % (antares_ir.replace('"', '\\"'), input_dict)

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
  meta_bgn = source.index('///') + len('///')
  meta_pos = source.index(':', meta_bgn)
  meta_end = source.index('\n', meta_pos)
  meta_inputs = source[meta_bgn:meta_pos].split(',')
  meta_outputs = source[meta_pos + 1:meta_end].split(',')

  code_name = 'Antares' + expr_hash
  source_path = '/tmp/antares_tf_%s.cc.kernel.cu' % code_name

  # Compile Kernel object
  with open(source_path, 'w') as fp:
    fp.write(source)
  return source, source_path, expr_hash, meta_inputs, meta_outputs

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

  def forward(self, antares_ir, inputs):
    antares_expr = generate_antares_expression(antares_ir, inputs)

    expr_hash = hashlib.sha256(antares_expr.encode()).hexdigest()
    if expr_hash in self.ops:
      attributes = self.ops[expr_hash]
    else:
      attributes = fetch_and_compile_antares_kernel(antares_expr, expr_hash, self.server_addr)
      self.ops[expr_hash] = attributes

    outputs = antares_custom_op.forward(inputs, *attributes)
    return outputs
