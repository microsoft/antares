# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import hashlib
import numpy as np
import copy

from .auto_shard import update_ast

def get_execution_parallism():
  return 1

def to_search_space(ast_seq, input_dict, output_dict):
  assert len(ast_seq) == 1, "Unimplemented multi ASTs."
  ast_props = ast_seq[0]['props']

  space = dict([('tile_%d' % i, {'_type': 'factor', '_value': [x['range'], 3]}) for i, x in enumerate(ast_props['data_axes'])])
  return space


def to_kernel_slices(compute_graph, best_config):
  ast_seq, input_dict, output_dict = copy.deepcopy(compute_graph)
  update_ast(best_config, ast_seq, input_dict, output_dict)

  assert len(ast_seq) == 1, "Unimplemented multi ASTs."
  ast = ast_seq[0]
  input_dict = ast['props']['input_dict']
  data_axes = ast['props']['data_axes']
  output = (ast['root'].dtype(), ast['props']['output_name'], {'dtype': ast['root'].dtype(), 'shape': [x['range'] for x in ast['props']['data_axes']]})
  curr_config = [best_config.get(f"tile_{i}", [-1, 1, 1]) for i in range(len(data_axes))]
  reduce_type = ast['props']['reduce_type']

  def indent(level):
    return "  " * (level + 1)

  def aw_stride(locations, access_shape):
    stride = [1] * len(access_shape)
    for i in reversed(range(len(access_shape) - 1)):
      stride[i] = stride[i + 1] * access_shape[i + 1]
    return ' + '.join([f'{l} * {s}' if s != 1 else f'{l}' for l, s in zip(locations, stride)])

  def get_data_rank(ax_name):
    for i, x in enumerate(data_axes):
      if x['name'] == ax_name:
        return i
    return -1

  def express(root):
    if root._op == 'op':
      inputs = root._value["inputs"]
      if len(inputs) == 2:
        return f'({express(inputs[0])} {root._value["name"]} {express(inputs[1])})'
      else:
        raise Exception('Unhandled inputs size in AST: %s' % inputs)
    elif root._op == 'get_item':
      return f'{root._value["tensor"]._value}[{aw_stride([express(x) for x in root._value["index"]], input_dict[root._value["tensor"]._value]["shape"])}]'
    elif root._op == 'axis':
      rank = get_data_rank(root._value)
      if rank >= 0:
        return f'({root._value}_outer * {curr_config[rank][2]} + {root._value}_inner)'
      else:
        return root._value
    elif root._op == 'const':
      if root.dtype() == 'float32':
        return str(root._value) + 'f'
      return str(root._value)
    elif root._op == 'cast':
      return f'(({root.dtype()}){express(root._value["inputs"][0])})'
    elif root._op == 'axis_range':
      for x in data_axes + ast['props']['reduce_axes']:
        if x['name'] == root._value:
          return str(x['range'])
      assert False
    elif root._op == 'when':
      relation = " && " if root._value['merge_op'] == 'all' else " || "
      return f'({relation.join([express(x) for x in root._value["if"]])} ? {express(root._value["true"])} : {express(root._value["false"])})'
    elif root._op == 'call':
      return f'{root._value["name"]}({", ".join([express(x) for x in root._value["inputs"]])})'
    else:
      raise Exception('Unhandled Op type in AST: %s' % root._op)

  body, num_locals = [], 1
  for i, x in enumerate(data_axes):
    body.append(f'{indent(i)}for (int {x["name"]}_outer = 0; {x["name"]}_outer < {curr_config[i][1]}; ++{x["name"]}_outer) {{')
    num_locals *= curr_config[i][2]

  if reduce_type:
    body.append(f'{indent(len(data_axes))}{output[0]} {output[1]}_local[{num_locals}] = {{}};\n')
    for i, x in enumerate(ast['props']['reduce_axes']):
      body.append(f'{indent(len(data_axes) + i)}for (int {x["name"]} = 0; {x["name"]} < {x["range"]}; ++{x["name"]})')
    base_level = len(ast['props']['reduce_axes'])
  else:
    base_level = 0

  inner_outputs, inner_shapes = [], []
  for i, x in enumerate(data_axes):
    body.append(f'{indent(i + len(data_axes) + base_level)}#pragma unroll\n{indent(i + len(data_axes) + base_level)}for (int {x["name"]}_inner = 0; {x["name"]}_inner < {curr_config[i][2]}; ++{x["name"]}_inner) {{')
    inner_outputs.append(f'{x["name"]}_inner')
    inner_shapes.append(curr_config[i][2])

  if reduce_type:
    if reduce_type == '+':
      body.append(f'{indent(len(data_axes) * 2 + base_level)}{output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}] {reduce_type}= {express(ast["root"])};')
    elif reduce_type == '>':
      body.append(f'{indent(len(data_axes) * 2 + base_level)}{output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}] = max({output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}], {express(ast["root"])});')
    elif reduce_type == '<':
      body.append(f'{indent(len(data_axes) * 2 + base_level)}{output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}] = min({output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}], {express(ast["root"])});')
    else:
      raise Exception('Unhandled reduce type in AST: %s' % reduce_type)
  else:
    body.append(f'{indent(len(data_axes) * 2 + base_level)}{output[1]}[{aw_stride(["(%s_outer * %s + %s_inner)" % (x["name"], curr_config[i][2], x["name"]) for i, x in enumerate(data_axes)], [x["range"] for i, x in enumerate(data_axes)])}] = {express(ast["root"])};')

  for i, x in enumerate(data_axes):
    body.append(f'{indent(len(data_axes) * 2 + base_level - 1 - i)}}}')

  if reduce_type:
    for i, x in enumerate(data_axes):
      body.append(f'{indent(i + len(data_axes))}#pragma unroll\n{indent(i + len(data_axes))}for (int {x["name"]}_inner = 0; {x["name"]}_inner < {curr_config[i][2]}; ++{x["name"]}_inner)')
    body.append(f'{indent(len(data_axes) * 2)}{output[1]}[{aw_stride(["(%s_outer * %s + %s_inner)" % (x["name"], curr_config[i][2], x["name"]) for i, x in enumerate(data_axes)], [x["range"] for i, x in enumerate(data_axes)])}] = {output[1]}_local[{aw_stride(inner_outputs, inner_shapes)}];')

  for i, x in enumerate(data_axes):
    body.append(f'{indent(len(data_axes) - 1 - i)}}}')

  body = '  \n'.join(body)

  return [
    [0, 'compute_%s' % ast['props']['output_name'], [(input_dict[x]['dtype'], x, input_dict[x]) for x in input_dict] + [output], body],
  ]


def do_native_translation_v2(codeset, **kwargs):
  if 'einstein_v2' not in os.environ['COMPUTE_V1']:
    raise Exception("Program for graphcore must be based on Antares IR")

  kernel_name, in_args, out_args, body = codeset

  func_args, delta_args = '', []
  for buf in in_args:
    if buf[1].startswith('_'):
      delta_args.append(buf[1])
      continue
    func_args += ' Input<Vector<%s>> %s;\n' % (buf[0], buf[1])
  for buf in out_args:
    func_args += ' Output<Vector<%s>> %s;\n' % (buf[0], buf[1])

  blend_code = getattr(kwargs['attrs'], 'blend', '').strip()
  blend_code = 'namespace {\n%s\n}\n\n' if blend_code else ''

  from antares.common import local_get_dir_file
  with open(local_get_dir_file('range_book.json'), 'r') as fp:
    range_book = json.load(fp)

  props = []
  for k in range_book['book']:
    arr2d = range_book['book'][k]
    arr2d = [str(x)[1:-1].replace(', ', ',') for x in arr2d]
    arr2d = '/'.join(arr2d)
    props.append(k + '/' + arr2d)
  props = ';'.join(props)

  full_body = f'''// Antares Property (k * ax_id + l .. r): {props}

#include <poplar/Vertex.hpp>

using namespace poplar;

#define int8 char
#define int16 short
#define int32 int
#define int64 long
#define float16 half
#define float32 float
#define float64 double

#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

{blend_code}
class CODELET_{kernel_name}: public Vertex {{

public:
 bool compute() {{
{body}
  return true;
 }}

{func_args}}};
'''
  return full_body
