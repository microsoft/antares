# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess
import json
import os

def _schedule_single(attrs, output, op_name, tail_op):
  s = attrs.scheduler

  def cache_local(output):
    if not tail_op:
      OL = s.cache_write(output, 'local')
    else:
      s[output].set_scope('local')
      OL, output = output, s.outputs[0].output(0)
    return output, OL
  s.cache_local = cache_local

  num_inputs = len(s[output].op.input_tensors)

  # Rough classification of computing features
  if len(output.op.reduce_axis) > 0 and (num_inputs <= 1 or len(output.shape) <= 1):
    from .algo_reduce import schedule_branch
    return schedule_branch(attrs, output, f"R{op_name}:", tail_op)

  if len(output.op.reduce_axis) > 0:
    from .algo_tiling import schedule_branch
    return schedule_branch(attrs, output, f"T{op_name}:")

  from .algo_format import schedule_branch
  return schedule_branch(attrs, output, f"F{op_name}:")

def schedule(attrs):
  config = os.environ.get('CONFIG', '').strip()
  step = int(os.environ.get('STEP', '0'))
  attrs.advanced_sched = config or step > 0
  tail_op, explicit_ops = None, [x for x in attrs.explicit_ops]

  if (len(explicit_ops) > 1 and
      not explicit_ops[-1].output(0).op.reduce_axis):
    fuse_tail = attrs.auto_config.define_knob(f"FU", [False, True])
    tail_op = explicit_ops[-1]
    if fuse_tail:
      tail_op.is_fused, explicit_ops = True, explicit_ops[:-1]
    else:
      tail_op.is_fused = False

  for rank, op in enumerate(reversed(explicit_ops)):
    _schedule_single(attrs, op.output(0), op.name, tail_op if rank == 0 else None)

