# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import autotvm
from tvm import te, tir, target
import logging
import sys, time, subprocess
import json
import os
import importlib
import traceback
import re

from antares.common import Mock, AntaresGlobal, backend

def einstein_v2(exprss, input_dict, extra_outputs=[], **kwargs):
  ir = os.environ.get('LL_IR', '')
  if not ir:
    for k in input_dict:
     if len(input_dict[k]['shape']) == 0:
       input_dict[k]['shape'] = [1]
    from lang import einstein_v2
    ir = einstein_v2.emit_tvm_ir(exprss, input_dict, extra_outputs)
    assert(len(ir) > 0)
    os.environ['LL_IR'] = ir
    print('\n[LL-IR]\n%s\n' % ir)

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
  AntaresGlobal.local_arg_pros['_in'] += [{'name': name, 'dtype': dtype, 'shape': shape}]

  global placeholders
  if len(shape) == 0:
    shape = [1]
  placeholders[name] = te.placeholder(shape, dtype=cast_dtype(dtype), name=name)
  return placeholders[name]

def loop(length, start=0):
  return te.reduce_axis((start, length))

def output(shape, func=None, flops=None, name='output0', topi=None, dtype=None, tag='', final_output=True):
  if len(shape) == 0:
    shape = [1]
  if flops is None:
    flops = np.product(shape)
  if topi is not None:
    result = te.compute(topi.shape, lambda *X: topi[X], name=name, tag='')
  else:
    result = te.compute(shape, func, name=name, tag=tag)
  if not final_output:
    return result

  if not shape:
    shape = result.shape
  if not dtype:
    dtype = result.dtype
  target = {'name': name, 'shape': shape, 'dtype': dtype}
  AntaresGlobal.local_arg_pros['_out'].append(target)

  global output_saver
  output_saver["outputs"].append(result)
  output_saver["flops"] += flops
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
    _traverse(final_op)

    if not explicit_ops:
      explicit_ops.append(final_op)
    callback(explicit_ops[-1], explicit_ops)

def do_native_scheduling(attrs):

  def select_plan(plan_name):
    if plan_name.find('.') < 0:
      plan_name = 'standard.' + plan_name
    import importlib
    schedule_lib = importlib.import_module('platforms.%s.schedule.%s' % (attrs.backend, plan_name), __loader__.name)
    schedule_lib.schedule(attrs)
    return attrs.scheduler, attrs.inputs + attrs.outputs

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
  return select_plan(plan)


intermediate_output = 'MultipleOutputsTempVar'

def refactor_multiple_names(code, global_arg_props):
  if len(global_arg_props['_out']) <= 1:
    return code
  for i in range(len(global_arg_props['_out'])):
    std_name = global_arg_props['_out'][i]['name']
    code = re.sub(r'\b%s\b' % std_name, '__%s' % std_name, code)
    code = re.sub(r'\b%s_v%d\b' % (intermediate_output, i), std_name, code)
  return code

@autotvm.template("template_op")
def get_template_op(**kwargs):
  if 'COMPUTE_V1' not in os.environ:
    raise Exception("Environment variable `COMPUTE_V1` is not set")
  program = os.environ['COMPUTE_V1'].strip()
  assert(program.startswith('- '))

  global placeholders, output_saver
  placeholders, output_saver = {}, {"outputs": [], "flops": 0}
  AntaresGlobal.local_arg_pros = {'_in': [], '_out': []}

  program = program[2:].strip()
  if program:
    exec('import tvm; from tvm import topi; ' + program, globals())
    AntaresGlobal.local_arg_pros['_in'].sort(key=lambda x: x['name'])
    AntaresGlobal.local_arg_pros['_out'].sort(key=lambda x: x['name'])

    inputs = sorted(list(placeholders.values()), key=lambda x: x.name)
    outputs = sorted(output_saver["outputs"], key=lambda x: x.op.name)
    cfg = autotvm.get_config()
    cfg.flop = output_saver["flops"]

    anno, options = program.find('## @'), []
    if anno >= 0:
      program, options = program[:anno].strip(), program[program.index(':', anno) + 1:].strip().split('|')

    if len(outputs) > 1:
      def to_list(shape):
        return [int(d) for d in shape]
      for i in range(1, len(outputs)):
        assert to_list(outputs[0].shape) == to_list(outputs[i].shape), "Shape sizes for multiple outputs should be equal."
      outputs = te.compute(outputs[0].shape, lambda *X: [v[X] for v in outputs], name=intermediate_output)
    sch = te.create_schedule([outputs[i].op for i in range(len(outputs))])

    def _callback(op, explicit_ops):
        attrs = Mock()
        attrs.inputs = inputs
        attrs.explicit_ops = explicit_ops
        attrs.outputs = [op.output(0)]
        attrs.scheduler = sch
        attrs.auto_config = cfg
        attrs.backend = backend
        attrs.ir = program
        attrs.options = options
        attrs.blend = ''
        attrs.get_extent = lambda axis: int(str(axis).split('ext=')[-1].split(')')[0])

        AntaresGlobal.attrs = attrs
        do_native_scheduling(attrs)

    traverse_inline(sch, outputs[0].op, _callback)
    return sch, list(inputs) + list(outputs)
