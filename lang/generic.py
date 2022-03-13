# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te, tir, target
import logging
import sys, time, subprocess
import json
import os
import importlib
import traceback
import re

from antares.common import Mock, AntaresGlobal, backend, AutoConfig

def einstein_v2(exprss, input_dict, extra_outputs=[], **kwargs):
  if 'comments' in kwargs:
    os.environ['COMMENTS'] = json.dumps(kwargs['comments'])

  ir = os.environ.get('LL_IR', '')
  if not ir:
    for k in input_dict:
     if len(input_dict[k]['shape']) == 0:
       input_dict[k]['shape'] = [1]
    from lang import einstein_v2
    ir = einstein_v2.ir_graph_parser(exprss, input_dict, extra_outputs)
    assert(len(ir) > 0)
    os.environ['LL_IR'] = ir
    # print('\n[LL-IR]\n%s\n' % ir[ir.find('; ') + 2:])

  exec(ir, globals())

placeholders, output_saver = None, None
custom_dtypes = {"@": 0}

def args(name):
  global placeholders
  assert(name in placeholders)
  return placeholders[name]

def cast_dtype(dtype):
  idx = dtype.find('@')
  if idx < 0:
    return dtype

  # Update register list
  global custom_dtypes
  typename = dtype[:idx]
  if typename not in custom_dtypes:
    if len(custom_dtypes) <= 4:
      dtype_code = 150 + len(custom_dtypes)
    else:
      oldest = min(filter(lambda x: x != '@', custom_dtypes.keys()), key=lambda x: custom_dtypes[x][1])
      dtype_code = custom_dtypes[oldest][0]
      custom_dtypes.pop(oldest)
  else:
    dtype_code = custom_dtypes[typename][0]

  custom_dtypes[typename] = (dtype_code, custom_dtypes["@"], dtype)
  custom_dtypes["@"] += 1
  target.datatype.register(typename, dtype_code)

  bits = int(dtype[idx + 1:])
  if bits % 32 == 0:
    return 'custom[%s]32' % (dtype[:idx])
  else:
    raise Exception("Not support custom dtype of bits = %d" % bits)

def common_reduce(name, args=(0,)):
  if not isinstance(args, tuple) and not isinstance(args, list):
    args = (args, )
  def reduce_op(x, y):
    assert x.dtype == y.dtype , "Reduing elements that don't have same data type: %s v.s. %s" % (x.dtype, y.dtype)
    return tir.call_pure_extern(x.dtype, name, x, y, *args[1:])
  return te.comm_reducer(reduce_op, lambda t: tir.const(args[0], dtype=t), name=name)

def input(name, shape, dtype="float32"):
  global placeholders
  if len(shape) == 0:
    shape = [1]
  placeholders[name] = te.placeholder(shape, dtype=cast_dtype(dtype), name=name)
  return placeholders[name]

def loop(length, start=0, name=None):
  if name is not None:
    return te.reduce_axis((start, length), name=name)
  else:
    return te.reduce_axis((start, length))

def output(shape, func=None, name='output0', topi=None, dtype=None, tag='', final_output=True):
  if len(shape) == 0:
    shape = [1]
  if topi is not None:
    result = te.compute(topi.shape, lambda *X: topi[X], name=name, tag='')
  else:
    result = te.compute(shape, func, name=name, tag=tag)
  if not final_output:
    return result

  global output_saver
  output_saver["outputs"].append(result)
  return result

def traverse_inline(s, final_op, callback):
    visited = set()
    explicit_ops = []

    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        for tensor in op.input_tensors:
          if isinstance(tensor.op, te.tensor.ComputeOp):
            _traverse(tensor.op)
        if op.reduce_axis:
          explicit_ops.append(op)
        elif op not in s.outputs:
          s[op].compute_inline()
        else:
          explicit_ops.append(op)

    _traverse(final_op)
    callback(explicit_ops)

def do_native_scheduling(attrs):

  def select_plan(plan_name):
    if plan_name.find('.') < 0:
      plan_name = 'standard.' + plan_name
    schedule_lib = importlib.import_module('backends.%s.schedule.%s' % (attrs.backend, plan_name), __loader__.name)
    schedule_lib.schedule(attrs)

  plan = 'default'
  for opt in attrs.options:
    if opt.startswith('plan/'):
      for plan_name in opt[5:].split(','):
        idx = plan_name.find('=')
        if idx >= 0:
          backend, name = plan_name.split('=')
          plan = None
          if backend != attrs.backend:
            continue
          plan = name
        else:
          plan = plan_name
        break
  if plan is None:
    raise Exception("No available plan configured for backend: %s" % attrs.backend)
  try:
    return select_plan(plan)
  except ModuleNotFoundError:
    traceback.print_exc()
    # setattr(AntaresGlobal, 'mode', 'antares')
    return None


intermediate_output = 'MultipleOutputsTempVar'


def refactor_builtins(code):
  result_lines = []
  for line in code.split('\n'):
    at = re.search(r'\b__builtin_set\(', line)
    while at is not None:
      arg_list = []
      start, stop, cnt = at.start(), at.end(), 0
      for i in range(stop, len(line)):
        if line[i] in ('(', '['):
          cnt += 1
        elif line[i] in (')', ']'):
          cnt -= 1
        if cnt <= 0 and line[i] in (',', ')'):
          arg_list.append(line[stop:i].strip())
          stop = i + 1
          if line[i] == ')':
            line = line[:start] + f'(({arg_list[0]}) = ({arg_list[1]}))' + line[stop:]
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
      code = re.sub(fr'\b{intermediate_output}\b', std_name, code)
    else:
      code = re.sub(fr'\b{intermediate_output}{i}\b', std_name, code)
  return code

def get_template_op(**kwargs):
  if 'COMPUTE_V1' not in os.environ:
    raise Exception("Environment variable `COMPUTE_V1` is not set")
  program = os.environ['COMPUTE_V1'].strip()
  assert program.startswith('- '), "The computing expression doesn't start with proper prefix: - ..."

  global placeholders, output_saver
  placeholders, output_saver = {}, {"outputs": []}

  program = program[2:].strip()
  if program:
    exec('import tvm; ' + program, globals())
    # exec('import tvm; from tvm import topi; ' + program, globals())

    inputs = sorted(list(placeholders.values()), key=lambda x: x.name)
    outputs = sorted(output_saver["outputs"], key=lambda x: x.op.name)

    anno, options = program.find('## @'), []
    if anno >= 0:
      program, options = program[:anno].strip(), program[program.index(':', anno) + 1:].strip().split('|')

    if len(outputs) > 1:
      def to_list(shape):
        return [int(d) for d in shape]
      for i in range(1, len(outputs)):
        assert to_list(outputs[0].shape) == to_list(outputs[i].shape), "Shape sizes for multiple outputs should be equal: %s v.s. %s" % (to_list(outputs[0].shape), to_list(outputs[i].shape))
      outputs = te.compute(outputs[0].shape, lambda *X: [v[X] for v in outputs], name=intermediate_output)
    sch = te.create_schedule([outputs[i].op for i in range(len(outputs))])

    def get_device_props():
      props = tvm.runtime.ndarray.gpu(0)
      with open('%s/device_properties.cfg' % os.environ['ANTARES_DRIVER_PATH'], 'r') as fp:
        mem_bandwith = []
        while True:
          line = fp.readline()
          if not line:
            break
          key, val = line.split(': ')
          if key in ('GlobalMemoryBusWidth', 'MemoryClockRate'):
            mem_bandwith.append(float(val))
        mem_bandwith = 'inf' if not mem_bandwith else np.product(mem_bandwith) * 2.5e-7
        props.mem_bandwith = float(mem_bandwith)
      return props

    if not hasattr(AntaresGlobal, 'auto_config'):
      AntaresGlobal.auto_config = AutoConfig()

    def _callback(explicit_ops):
      attrs = Mock()
      attrs.device_props = get_device_props()
      attrs.inputs = list(inputs)
      attrs.outputs = list(outputs)
      attrs.explicit_ops = explicit_ops
      attrs.scheduler = sch
      attrs.auto_config = AntaresGlobal.auto_config
      attrs.backend = backend
      attrs.ir = program
      attrs.options = options
      attrs.blend = ''
      attrs.get_extent = lambda axis: int(axis.dom.extent)

      def get_lower():
        return str(tvm.lower(sch, attrs.inputs + attrs.outputs, simple_mode=True)).split('#[metadata]')[0]

      attrs.get_lower = get_lower
      AntaresGlobal.attrs = attrs
      do_native_scheduling(attrs)

    traverse_inline(sch, outputs[0].op, _callback)
    return sch, AntaresGlobal.attrs.inputs + AntaresGlobal.attrs.outputs
