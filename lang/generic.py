# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys, time, subprocess
import json
import os
import importlib
import traceback
import re
import copy

from antares.common import AntaresGlobal

def einstein_v2(exprss, input_dict, extra_outputs=[], **kwargs):
  for k in input_dict:
    if isinstance(input_dict[k], str):
      dtype, shapes = input_dict[k].strip()[:-1].split('[')
      if not shapes.strip():
        shapes = []
      else:
        shapes = [x.strip() for x in shapes.split(',')]
        shapes = [int(x) if x.isdigit() else x for x in shapes]
      input_dict[k] = {"dtype": dtype.strip(), "shape": shapes}

  if 'comments' in kwargs:
    os.environ['COMMENTS'] = json.dumps(kwargs['comments'])

  for k in input_dict:
   if len(input_dict[k]['shape']) == 0:
     input_dict[k]['shape'] = [1]

  from lang import einstein_v2
  ast_seq, input_dict, output_dict = einstein_v2.ir_graph_parser(exprss, input_dict, extra_outputs)

  # Registry Global Argument Properties
  global_arg_props = {'_in': [], '_out': []}
  for k in input_dict:
    prop = copy.deepcopy(input_dict[k])
    prop['name'] = k
    if f'___{k}' in output_dict:
      global_arg_props['_out'].append(prop)
    else:
      global_arg_props['_in'].append(prop)
  for k in output_dict:
    if k.startswith('___'):
      continue
    prop = copy.deepcopy(output_dict[k])
    prop['name'] = k
    global_arg_props['_out'].append(prop)
  global_arg_props['_in'].sort(key=lambda x: x['name'])
  global_arg_props['_out'].sort(key=lambda x: x['name'])

  AntaresGlobal.global_arg_props = global_arg_props
  AntaresGlobal.compute_graph = ast_seq, input_dict, output_dict

def implement_builtins(name, args):
  if name == '__builtin_set':
    assert len(args) == 2
    return f'(({args[0]}) = ({args[1]}))'
  elif name == '__builtin_add':
    assert len(args) == 2
    return f'atomicAdd(&({args[0]}), ({args[1]}))'
  else:
    raise Exception(f'Builtin function with {name} is not handled.')

def refactor_builtins(code):
  result_lines = []
  for line in code.split('\n'):
    at = re.search(r'\b__builtin_[a-z]+\(', line)
    while at is not None:
      start, stop, cnt = at.start(), at.end(), 0
      bname, arg_list = line[start:stop-1], []
      for i in range(stop, len(line)):
        if line[i] in ('(', '['):
          cnt += 1
        elif line[i] in (')', ']'):
          cnt -= 1
        if cnt <= 0 and line[i] in (',', ')'):
          arg_list.append(line[stop:i].strip())
          stop = i + 1
          if line[i] == ')':
            line = line[:start] + implement_builtins(bname, arg_list) + line[stop:]
            break
      at = re.search(r'\b__builtin_set\(', line)
    result_lines.append(line)
  return '\n'.join(result_lines)

def refactor_special_names(code, global_arg_props):
  code = code.replace('(int* __restrict__ _id, ', '(').replace('_id[(0)]', '_id')
  for i in range(len(global_arg_props['_out'])):
    std_name = global_arg_props['_out'][i]['name']
    code = re.sub(fr'\b___{std_name}\[.*\] = \b', '', code)

  code = refactor_builtins(code)
  if len(global_arg_props['_out']) <= 1:
    return code
  for i in range(len(global_arg_props['_out'])):
    std_name = global_arg_props['_out'][i]['name']
    code = re.sub(fr'\b{std_name}\b', f'__{std_name}', code)
    if i == 0:
      code = re.sub(fr'\bMultipleOutputsTempVar\b', std_name, code)
    else:
      code = re.sub(fr'\bMultipleOutputsTempVar{i}\b', std_name, code)
  return code

def load_template_op():
  if 'COMPUTE_V1' not in os.environ:
    raise Exception("Environment variable `COMPUTE_V1` is not set")
  program = os.environ['COMPUTE_V1'].strip()
  assert program.startswith('- '), "The computing expression doesn't start with proper prefix: - ..."
  program = program[2:].strip()
  exec(program)
