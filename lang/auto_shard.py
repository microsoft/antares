# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import numpy as np
from common import backend

def compute_local_ranges(nodes, ranges, input_dict, sliced_shape):
  tensor_slices = {}

  def _merge_range(l, r):
    if '*' in [l, r]:
      return '*'
    return [min(l[0], r[0]), max(l[1], r[1])]

  def _compute_slices(node):
    if node._op == 'const':
      val = int(node._value)
      return [val, val]
    elif node._op == 'get_item':
      tensor = node._value['tensor']
      index = node._value['index']
      slices = []
      for i, it in enumerate(index):
        subrange = _compute_slices(it)
        slices.append(subrange)
      if tensor._value['name'] in tensor_slices:
        prev = tensor_slices[tensor._value['name']]
        slices = [_merge_range(slices[i], prev[i]) for i in range(len(slices))]
      if tensor._value['name'] in input_dict:
        tensor_slices[tensor._value['name']] = slices
        if tensor._value['name'] not in sliced_shape:
          sliced_shape[tensor._value['name']] = [0] * len(slices)
        for i, rng in enumerate(slices):
          sliced_shape[tensor._value['name']][i] = max(sliced_shape[tensor._value['name']][i], rng[1] - rng[0] + 1) if rng != '*' else input_dict[tensor._value['name']]['shape'][i]
      return '*'
    elif node._op == 'axis':
      ax_name = node._value['name']
      if ax_name in ranges:
        return ranges[ax_name]
      return [0, node._value['range'] - 1]
    elif node._op == 'op':
      subrange = []
      for i in range(len(node._value["inputs"])):
        subrange.append(_compute_slices(node._value["inputs"][i]))
      if '*' in subrange:
        return '*'
      if node._value['name'] == '+' and len(subrange) == 2:
        return [subrange[0][0] + subrange[1][0], subrange[0][1] + subrange[1][1]]
      elif node._value['name'] == '-' and len(subrange) == 2:
        return [subrange[0][0] - subrange[1][1], subrange[0][1] - subrange[1][0]]
      elif node._value['name'] == '*' and len(subrange) == 2:
        d1, d2 = subrange[0][0] * subrange[1][0], subrange[0][0] * subrange[1][1]
        d3, d4 = subrange[0][1] * subrange[1][0], subrange[0][1] * subrange[1][1]
        return [min(d1, d2, d3, d4), max(d1, d2, d3, d4)]
      elif node._value['name'] == '<=' and len(subrange) == 2:
        if subrange[0][1] <= subrange[1][0]:
          return [1, 1]
        if subrange[0][0] >= subrange[1][1]:
          return [0, 0]
        return [0, 1]
      elif node._value['name'] == '<' and len(subrange) == 2:
        if subrange[0][1] < subrange[1][0]:
          return [1, 1]
        if subrange[0][0] > subrange[1][1]:
          return [0, 0]
        return [0, 1]
      else:
        raise Exception("TODO: Unhandled case - constant prop for other const ops `%s`" % node._value['name'])
    elif node._op == 'cast':
      return _compute_slices(node._value["inputs"][0])
    elif node._op == 'call':
      _compute_slices(node._value["inputs"][0])
      return '*'
    elif node._op == 'when':
      infer_fa = [0, 0]
      for cond in node._value['if']:
        infer_res = _compute_slices(cond)
        if infer_res == [0, 0]:
          infer_fa[0] += 1
        if infer_res == [0, 1]:
          infer_fa[1] += 1
      l, r = _compute_slices(node._value['true']), _compute_slices(node._value['false'])
      if infer_fa[0] > 0:
        subrange = r
      elif infer_fa[1] > 0:
        subrange = _merge_range(l, r)
      else:
        subrange = 1
      return subrange
    else:
      raise Exception('Unrecognized node type: %s' % node._op)

  for node in nodes:
    _compute_slices(node)
  return tensor_slices


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

  slice_results = []
  num_parallel = int(np.product(pieces))
  if num_parallel > 4096:
    raise Exception("Please be cautious of the whole number of parallelism: %d" % num_parallel)
  stride = []
  for i in range(len(data_axes)):
    size = data_axes[i]['range']
    assert(size % pieces[i] == 0)
    stride.append(size // pieces[i])

  sliced_shape = {}
  for i in range(num_parallel):
    step_id = i
    startup = []
    for j in range(len(pieces)):
      startup.append(step_id % pieces[j])
      step_id //= pieces[j]
    ranges = {}
    for j in range(len(data_axes)):
      ax_name = data_axes[j]['name']
      ranges[ax_name] = [stride[j] * startup[j], stride[j] * (startup[j] + 1) - 1]
    roots = [ast['root']] + ([ast['injective']['root']] if 'injective' in ast else [])
    tensor_slices = compute_local_ranges(roots, ranges=ranges, input_dict=ast['props']['input_dict'], sliced_shape=sliced_shape)
    slice_results.append((ranges, tensor_slices))

  for i in range(len(data_axes)):
    data_axes[i]['range'] //= pieces[i]

  ast['props']['slices'] = slice_results
  for k in ast['props']['input_dict']:
    if k in sliced_shape:
     ast['props']['input_dict'][k]['shape'] = sliced_shape[k]

  if 'injective' in ast:
    ast['injective']['props']['slices'] = ast['props']['slices']
    ast['injective']['props']['input_dict'] = ast['props']['input_dict']
    assert(id(data_axes) == id(ast['injective']['props']['data_axes']))
