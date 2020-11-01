# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import numpy as np
from common import backend
from lang.einstein_v2 import walk_in_ast


def infer_range(root, ax_rank):
  if root._op == 'get_item':
    return '*'
  if root._op == 'const':
    ival = int(root._value)
    return [0, None, ival, ival]
  if root._op == 'axis':
    if root._value['name'] in ax_rank:
      return [1, root._value['name'], 0, 0]
    else:
      return [0, None, 0, root._value['range'] - 1]
  if root._op == 'op':
    if root._value['name'] == '*' and len(root._value['inputs']) == 2:
      ll, rr = infer_range(root._value['inputs'][0], ax_rank), infer_range(root._value['inputs'][1], ax_rank)
      if ll == '*' or rr == '*':
        return '*'
      if rr[1] is not None:
        ll, rr = rr, ll
      if rr[1] is not None:
        return '*'
      a0, a1, a2, a3 = ll[2] * rr[2], ll[2] * rr[3], ll[3] * rr[2], ll[3] * rr[3]
      amin, amax = min(a0, a1, a2, a3), max(a0, a1, a2, a3)
      if ll[1] is None:
        return [ll[0], ll[1], amin, amax]
      elif rr[2] != rr[3]:
        return '*'
      else:
        return [ll[0] * rr[2], ll[1], amin, amax]
    if root._value['name'] == '+' and len(root._value['inputs']) == 2:
      ll, rr = infer_range(root._value['inputs'][0], ax_rank), infer_range(root._value['inputs'][1], ax_rank)
      if ll == '*' or rr == '*':
        return '*'
      if rr[1] is not None:
        ll, rr = rr, ll
      if rr[1] is not None:
        if ll[1] != rr[1]:
          return '*'
        return [ll[0] + rr[0], ll[1], ll[2] + rr[2], ll[3] + rr[3]]
      return [ll[0], ll[1], ll[2] + rr[2], ll[3] + rr[3]]
    if root._value['name'] == '-' and len(root._value['inputs']) == 2:
      ll, rr = infer_range(root._value['inputs'][0], ax_rank), infer_range(root._value['inputs'][1], ax_rank)
      if ll == '*' or rr == '*':
        return '*'
      if ll[1] == rr[1]:
        return [ll[0] - rr[0], ll[1], ll[2] - rr[3], ll[3] - rr[2]]
      if ll[1] == None:
        return [- rr[0], rr[1], ll[2] - rr[3], ll[3] - rr[2]]
      if rr[1] == None:
        return [ll[0], ll[1], ll[2] - rr[3], ll[3] - rr[2]]
      return '*'
    if root._value['name'] == '%' and len(root._value['inputs']) == 2:
      ll, rr = infer_range(root._value['inputs'][0], ax_rank), infer_range(root._value['inputs'][1], ax_rank)
      if ll == '*' or rr == '*':
        return '*'
      if rr[1] != None or rr[2] != rr[3]:
        return '*'
      return [0, None, 0, rr[2] - 1]

  raise Exception('Unhandled infer_range op type: %s' % root)

def scan_items(root, ast, range_book):
  if root._op != 'get_item':
    return

  ax_rank = {None: -1}
  for i, item in enumerate(ast['props']['data_axes']):
    ax_rank[item['name']] = i

  tensor_name = root._value['tensor']._value['name']
  current_range = []
  for i, sub in enumerate(root._value['index']):
    tensor_index = i
    index_range = infer_range(sub, ax_rank)
    if index_range == '*':
      index_range = [0, None, 0, ast['props']['data_axes'][i]['range'] - 1]
    index_range[1] = ax_rank[index_range[1]]
    current_range.append(index_range)

  if tensor_name in range_book:
    previous_range = range_book[tensor_name]
    assert len(previous_range) == len(current_range)
    for i in range(len(current_range)):
      if previous_range[i] != current_range[i]:
        current_range[i] = [0, None, 0, ast['props']['data_axes'][i]['range'] - 1]
  range_book[tensor_name] = current_range

def auto_shard_on_ast(ast):
  if backend not in ['c-gc']:
    return

  steps = int(os.environ.get('STEP', '0'))
  pieces = os.environ.get('CONFIG', '').strip()

  data_axes = ast['props']['data_axes']
  if not pieces and steps > 0:
    return

  try:
    pieces = json.loads(pieces)
    pieces = [pieces['axis_%d' % i][-1] for i in range(len(data_axes))]
  except:
    pieces = [1] * len(data_axes)

  assert 'injective' not in ast, "Unhandled injective case for graphcore."
  range_book = {}
  walk_in_ast(ast['root'], scan_items, [ast, range_book], ast, 'root')
  ast['props']['shard'] = {'nparts': pieces, 'book': range_book}

  # AST props: ast['props']['data_axes'], ast['props']['output_dict'], ast['props']['input_dict']
  for i in range(len(pieces)):
    assert data_axes[i]['range'] % pieces[i] == 0
    data_axes[i]['range'] //= pieces[i]
    for k in ast['props']['output_dict']:
      output_item = ast['props']['output_dict'][k]
      assert output_item['shape'][i] % pieces[i] == 0
      output_item['shape'][i] //= pieces[i]
  for k in ast['props']['input_dict']:
    input_item = ast['props']['input_dict'][k]
    sub_shape = []
    for it in range_book[k]:
      bias_diff = it[3] - it[2] + 1
      if it[1] < 0 or it[0] == 0:
        sub_shape.append(bias_diff)
      elif it[0] > 0:
        sub_shape.append(it[0] * (data_axes[it[1]]['range'] - 1) + bias_diff)
      else:
        raise Exception('Unhandled book case:', it)
    input_item['shape'] = sub_shape

  from antares.common import local_get_dir_file
  output_key = next(iter(ast['props']['output_dict']))
  ast['props']['shard']['local_shape'] = ast['props']['output_dict'][output_key]['shape']
  with open(local_get_dir_file('range_book.json'), 'w') as fp:
    json.dump(ast['props']['shard'], fp)
