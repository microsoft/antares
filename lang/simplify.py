# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import itertools
import numpy as np
from common import backend
from lang.einstein_v2 import walk_in_ast, OpTensor

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

def get_daxis_range(name, ast):
  for k in ast['props']['data_axes']:
    if k['name'] == name:
      return k['range']
  return -1

def get_input_shape(name, ast):
  return ast['props']['input_dict'][name]['shape']

def scan_items(root, ast, access_book, tensor_nodes):
  if root._op == 'axis':
    access_book['*'].append(root._value['name'])
  if root._op != 'get_item':
    return
  tensor_nodes.append(root)
  tensor_name = root._value['tensor']._value['name']
  tensor_index = []
  for i, sub in enumerate(root._value['index']):
    if sub._op == 'axis':
      rng = get_daxis_range(sub._value['name'], ast)
      shp = get_input_shape(tensor_name, ast)
      if rng == shp[i]:
        tensor_index.append(sub._value['name'])
      else:
        tensor_index.append('*')
    else:
      tensor_index.append('*')
      walk_in_ast(sub, scan_items, [ast, access_book, tensor_nodes], root._value['index'], i)
  if tensor_name in access_book:
    last_index = access_book[tensor_name]
    assert len(last_index) == len(tensor_index)
    for i in range(len(tensor_index)):
      if tensor_index[i] != last_index[i]:
        tensor_index[i] = '*'
  else:
    access_book[tensor_name] = tensor_index
  return ''

def update_ast_axis(ast, seq, tensor_nodes):
  new_axis = '_'.join(seq)
  new_data_axes = []
  for i, x in enumerate(ast['props']['data_axes']):
    if x['name'] == seq[0]:
      new_data_axes.append({'name': new_axis, 'range': int(np.product([v['range'] for v in ast['props']['data_axes'][i:i+len(seq)]]))})
    elif x['name'] in seq:
      continue
    else:
      new_data_axes.append(x)
  ast['props']['data_axes'] = new_data_axes
  for k in ast['props']['output_dict']:
    ast['props']['output_dict'][k]['shape'] = [x['range'] for x in new_data_axes]
    break

  input_shape_alter_info = {}
  for k in tensor_nodes:
    index = k._value['index']
    it, new_index = 0, []
    while it < len(index):
      dim = index[it]
      if dim._op != 'axis' or dim._value['name'] != seq[0]:
        new_index.append(dim)
        it += 1
      else:
        tensor_name = k._value['tensor']._value['name']
        if tensor_name not in input_shape_alter_info:
          input_shape_alter_info[tensor_name] = set()
        new_range = np.product([v._value['range'] for v in index[it:it+len(seq)]])
        item = OpTensor('axis', {"name": new_axis, "range": int(new_range)}, 'int32')
        new_index.append(item)
        input_shape_alter_info[tensor_name].add((it, it + len(seq), new_axis, new_range))
        it += len(seq)
    k._value['index'] = new_index

  for k in ast['props']['input_dict']:
    tensor_shape = ast['props']['input_dict'][k]
    if k not in input_shape_alter_info:
      continue
    for h in input_shape_alter_info[k]:
      l, r, _, rng = h
      tensor_shape['shape'] = tensor_shape['shape'][0:l] + [rng] + tensor_shape['shape'][r:]
    for t in tensor_nodes:
      t._value['tensor']._value['shape'] = tensor_shape

  ## print(ast)


def scan_trivial_axis(root, ast):
  if root._op == 'axis' and root._value['range'] == 1:
    return OpTensor('const', 0, 'int32')

def eliminate_trivial_axis(ast):
  print(ast['props'])
  walk_in_ast(ast['root'], scan_trivial_axis, [ast], ast, 'root')

  def update(axes, start=0):
    new_ra = axes[:start]
    for ax in axes[start:]:
      if ax['range'] != 1:
        new_ra.append(ax)
    axes = new_ra
    return axes

  ast['props']['reduce_axes'] = update(ast['props']['reduce_axes'])
  if not ast['props']['reduce_axes']:
    ast['props']['reduce_type'] = None
  for k in ast['props']['output_dict']:
    num_outputs = int(np.product(ast['props']['output_dict'][k]['shape']))
    break
  if num_outputs > 1:
    ast['props']['data_axes'] = update(ast['props']['data_axes'])
    for k in ast['props']['output_dict']:
      ast['props']['output_dict'][k]['shape'] = [x for x in filter(lambda x: x != 1, ast['props']['output_dict'][k]['shape'])]
  elif len(ast['props']['data_axes']) > 1: 
    ast['props']['data_axes'] = update(ast['props']['data_axes'], 1)
    for k in ast['props']['output_dict']:
      ast['props']['output_dict'][k]['shape'] = [1]
  # print(ast['props'])

def compute(ast):
  if os.environ.get('SIMPLE', '1') == '0':
    return
  if 'injective' in ast or 'shard' in ast['props']:
    # FIXME: Unhandled case yet
    return

  annotation = os.environ.get('COMPUTE_V1', '').split('##')[-1]
  # FIXME: Just a rough check
  if 'plan/' in annotation and 'default' not in annotation:
    return

  eliminate_trivial_axis(ast)

  access_book, tensor_nodes = {}, []
  access_book['*'] = []
  walk_in_ast(ast['root'], scan_items, [ast, access_book, tensor_nodes], ast, 'root')

  data_axes = ast['props']['data_axes']
  access_book['='] = [x['name'] for x in data_axes]
  unique_axes = set()
  for k in access_book:
    if k != '*':
      access_book[k] = [x if x not in access_book['*'] else '*' for x in access_book[k]]
      for x in access_book[k]:
        if x != '*':
          unique_axes.add(x)
  access_book.pop('*')
  # print(access_book, unique_axes)

  visited = set()
  for size in reversed(range(2, len(unique_axes) + 1)):
    for k in itertools.permutations(unique_axes, size):
      if sum([1 if x in visited else 0 for x in k]) > 0:
        continue
      this_pattern = ':%s:' % ':'.join(k)
      access_pattern = [':%s:' % ':'.join(access_book[x]) for x in access_book]
      can_simplify = True
      for acc in access_pattern:
        rest_acc = ''.join(acc.split(this_pattern)).split(':')
        if sum([1 if x in k else 0 for x in rest_acc]) > 0:
          can_simplify = False
      if can_simplify:
        for x in k:
          visited.add(x)
        update_ast_axis(ast, k, tensor_nodes)
      # print(k, this_pattern, access_pattern, can_simplify)
  return

