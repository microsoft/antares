# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import copy
import importlib
import json
import re

from antares.common import Mock, AutoConfig, AntaresGlobal, product, backend


def codegen(ast_seq, input_dict, output_dict, config, space_only=False):
  assert len(ast_seq) == 1
  ast = ast_seq[0]

  if not hasattr(AntaresGlobal, 'attrs'):
    AntaresGlobal.attrs = Mock()
    AntaresGlobal.attrs.blend = ''
    AntaresGlobal.attrs.backend = backend
    AntaresGlobal.auto_config = AutoConfig()

    def get_device_props():
      props = Mock()
      with open('%s/device_properties.cfg' % os.environ['ANTARES_DRIVER_PATH'], 'r') as fp:
        mem_bandwith, compute_version = [], '.'
        while True:
          line = fp.readline()
          if not line:
            break
          key, val = line.split(': ')
          val = val.strip()
          if key in ('GlobalMemoryBusWidth', 'MemoryClockRate'):
            mem_bandwith.append(float(val))
          elif key in ('ComputeCapabilityMajor'):
            compute_version = val + compute_version
          elif key in ('ComputeCapabilityMinor'):
            compute_version = compute_version + val
        mem_bandwith = 'inf' if not mem_bandwith else product(mem_bandwith) * 2.5e-7
        props.mem_bandwith = float(mem_bandwith)
        props.compute_version = compute_version
      return props

    AntaresGlobal.attrs.device_props = get_device_props()

  if space_only:
    space = dict([(x['name'], {'_type': 'factor', '_value': [8192 * 2, 4]}) for i, x in enumerate(ast['props']['data_axes'])])
    AntaresGlobal.auto_config._config = space
    return space

  vamap, vamap_count = json.loads(os.environ.get("VAMAP_EXTRA", {})), []
  vamap_axes = json.loads(os.environ['VAMAP_AXES'])
  if ast['props']['output_name'] not in vamap:
    vamap[ast['props']['output_name']] = [None] * len(ast['props']['data_axes'])

  def query_stride(name, idx):
    if vamap[name][idx] is not None:
      return vamap[name][idx]
    if name == ast['props']['output_name']:
      return ast['props']['data_axes'][idx]['range']
    return ast['props']['input_dict'][name]["shape"][idx]

  def merge_index(locations, name):
    stride = [1] * len(vamap[name])
    for i in reversed(range(len(vamap[name]) - 1)):
      stride_dim = str(query_stride(name, i + 1))
      stride[i] = f"{stride_dim} * {stride[i + 1]}" if stride[i + 1] != 1 else stride_dim

    return ' + '.join([f'{l} * ({s})' if str(s) != "1" else l for l, s in zip(locations, stride)])

  def express_left(props):
    for i, k in enumerate(props['data_axes']):
        if vamap[props['output_name']][i] is None and k['name'] in vamap_axes:
          vamap[props['output_name']][i] = '_' + vamap_axes[k['name']]
    return f"{props['output_name']}[{merge_index([x['name'] for x in props['data_axes']], props['output_name'])}]"

  def express(root):
    if root._op == 'op':
      inputs = root._value["inputs"]
      if len(inputs) == 2:
        return f'({express(inputs[0])} {root._value["name"]} {express(inputs[1])})'
      elif len(inputs) == 1:
        return f'({root._value["name"]}{express(inputs[0])})'
      else:
        raise Exception('Unhandled inputs size in AST: %s' % inputs)
    elif root._op == 'get_item':
      for i, x in enumerate(root._value["index"]):
        if x._op == 'axis':
          tensor_name = root._value["tensor"]._value
          guide_output_id = [r for r, ax in enumerate(ast['props']['data_axes']) if ax['name'] == x._value]
          if guide_output_id:
            if tensor_name in vamap and vamap[tensor_name][i] is not None:
              vamap[ast['props']['output_name']][guide_output_id[0]] = vamap[tensor_name][i]
            else:
              vamap[ast['props']['output_name']][guide_output_id[0]] = ast['props']['input_dict'][tensor_name]["shape"][i]
      return f'{root._value["tensor"]._value}[{merge_index([express(x) for x in root._value["index"]], root._value["tensor"]._value)}]'
    elif root._op == 'axis':
      return root._value
    elif root._op == 'const':
      if root.dtype() == 'float32':
        return str(root._value) + 'f'
      return str(root._value)
    elif root._op == 'cast':
      return f'(({root.dtype()}){express(root._value["inputs"][0])})'
    elif root._op == 'axis_range':
      for x in ast['props']['data_axes'] + ast['props']['reduce_axes']:
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

  # Must Express Right to infer left-side shape infos
  code = express(ast['root']);
  code = express_left(ast['props']) + ' = ' + code + ';';

  loops, d_axes, splits = '', ['x', 'y', 'z'], []
  for ax in ast['props']['data_axes']:
    splits += [(ax['name'], config.get(ax['name'], [-1, 1, 1, 1])[1:])]

  thread_strides = [sp[1] for _, sp in splits]
  total_threads = product(thread_strides)
  if total_threads > 1024:
    raise

  block_strides = [(ax['range'] + product(sp) - 1) // product(sp) for ax, (_, sp) in zip(ast['props']['data_axes'], splits)]
  loops += f'  // [thread_extent] blockIdx.x = {product(block_strides)}\n  int __tasks_block = blockIdx.x;\n'
  for i, (ax_name, sp) in enumerate(splits):
    loops += f'  int {ax_name}_0 = __tasks_block % {block_strides[i]} * {product(sp)}; __tasks_block /= {block_strides[i]};\n'
  loops += '\n'
  loops += f'  // [thread_extent] threadIdx.x = {total_threads}\n  int __tasks_thread = threadIdx.x;\n'
  for i, (ax_name, sp) in enumerate(splits):
    loops += f'  int {ax_name}_2 = __tasks_thread % {thread_strides[i]} * {product(sp[2:])}; __tasks_thread /= {thread_strides[i]};\n'
  loops += '\n'
  unroll_step = int(os.environ.get('UNROLL', 32))

  def codegen(bound_check):
    loops = ''

    for i, (ax_name, sp) in enumerate(splits):
      loops += f'  for (int {ax_name}_1 = 0; {ax_name}_1 < {product(sp[0:1])} * {product(sp[1:])}; {ax_name}_1 += {product(sp[1:])})\n'

    refactor_code, real_bounds = code, []
    for i, (ax_name, sp) in enumerate(splits):
      if product(sp[2:]) <= unroll_step:
        loops += '  #pragma unroll\n'
      loops += f'  for (int {ax_name}_3 = 0; {ax_name}_3 < {product(sp[2:])}; {ax_name}_3 ++)\n'
      real_name = f'({ax_name}_0 + {ax_name}_1 + {ax_name}_2 + {ax_name}_3)'
      refactor_code = re.sub(rf'\b{ax_name}\b', real_name, refactor_code)
      real_bounds += [real_name]
    if bound_check:
      loops += '    if (' + ' && '.join([f'{real_name} < {query_stride(ast["props"]["output_name"], i)}' for i, real_name in enumerate(real_bounds)]) + ')\n  '
    loops += '    ' + refactor_code
    return loops

  loops += 'if (' + ' && '.join([f'{ax_name}_0 + {product(sp)} <= {query_stride(ast["props"]["output_name"], i)}' for i, (ax_name, sp) in enumerate(splits)]) + ') {\n'
  loops += codegen(False)
  loops += '\n} else {\n'
  loops += codegen(True)
  loops += '\n}'

  code = loops + '\n'
  code = re.sub(r'\bfloat64\b', 'double', code)
  code = re.sub(r'\bfloat32\b', 'float', code)
  code = re.sub(r'\bfloat16\b', 'half', code)
  code = re.sub(r'\bint64\b', 'long', code)
  code = re.sub(r'\bint32\b', 'int', code)
  code = re.sub(r'\bint16\b', 'short', code)
  code = re.sub(r'\bint8\b', 'char', code)

  # Build VAMAP args to function
  vamap_args, visited = [], set()
  for key in vamap:
    for i, x in enumerate(vamap[key]):
      if isinstance(x, str) and x not in visited:
        visited.add(x)
        if key in ast['props']['input_dict']:
          value = ast['props']['input_dict'][key]["shape"][i]
        else:
          value = ast['props']['data_axes'][i]['range']
        vamap_args.append(f'{x}:{value}')
  vamap_args = sorted(vamap_args)
  if vamap_args:
    os.environ['VAMAP'] = ','.join(vamap_args)

  return [
    [0, 'kernel_%s' % ast['props']['output_name'], sorted([('float', x, ast['props']['input_dict'][x]) for x in ast['props']['input_dict']], key=lambda x: x[1]) + [
        ('float', ast['props']['output_name'], {'dtype': ast['root'].dtype(), 'shape': [x['range'] for x in ast['props']['data_axes']]})
      ], code],
  ]

