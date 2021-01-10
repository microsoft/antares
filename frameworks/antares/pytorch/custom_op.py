# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os, json, hashlib
from torch.autograd import Function
from http import client as http_client

import antares_custom_op

def generate_antares_expression(antares_ir, feed_dict, extra_outputs):
  input_dict, kwargs = {}, {}
  for k, v in feed_dict:
    input_dict[k] = {
    'dtype': str(v.dtype).split('.')[1],
    'shape': list(v.shape)
    }
    kwargs[k] = v

  input_dict = json.dumps(input_dict)
  return '- einstein_v2("%s", input_dict=%s, extra_outputs=%s)' % (antares_ir.replace('"', '`'), input_dict, extra_outputs)

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


class CustomOp(torch.nn.Module):
  def __init__(self, server_addr='localhost'):
    super(CustomOp, self).__init__()
    if server_addr.find(':') < 0:
      server_addr += ':8880'
    self.server_addr = server_addr
    self.ops = dict()

  def forward(self, ir, mapped_values, mapped_keys=[], extra_outputs=[]):
    keys, values = mapped_keys, mapped_values
    if not keys:
      keys = [f'input{i}' for i in range(len(values))]
    antares_expr = generate_antares_expression(ir, zip(keys, values), extra_outputs)

    expr_hash = hashlib.sha256(antares_expr.encode()).hexdigest()
    if expr_hash in self.ops:
      output_names, attributes = self.ops[expr_hash]
    else:
      output_names, attributes = fetch_and_compile_antares_kernel(antares_expr, expr_hash, self.server_addr)
      self.ops[expr_hash] = output_names, attributes

    outputs = antares_custom_op.forward(values, *attributes)
    for i in range(len(outputs)):
      outputs[i].id = output_names[i]

    if len(outputs) == 1:
       outputs = outputs[0]
    else:
       outputs = tuple(outputs)
    return outputs
