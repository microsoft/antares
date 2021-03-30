# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess
import json
import os

def _schedule_single(attrs, output, rank, have_tail):
  s = attrs.scheduler

  def cache_local(output):
    if not have_tail:
      OL = s.cache_write(output, 'local')
    else:
      s[output].set_scope('local')
      OL, output = output, s.outputs[0].output(0)
    return output, OL
  s.cache_local = cache_local

  num_inputs = len(s[output].op.input_tensors)

  # Rough classification of computing features
  if attrs.is_tuning and num_inputs > 1 and len(output.op.reduce_axis) > 0:
    from .algo_tiling import schedule_branch
    return schedule_branch(attrs, output, f"T{rank}:")

  if attrs.is_tuning and not have_tail and len(output.op.reduce_axis) > 0:
    from .algo_reduce import schedule_branch
    return schedule_branch(attrs, output, f"R{rank}:")

  from .algo_format import schedule_branch
  return schedule_branch(attrs, output, f"F{rank}:")

def schedule(attrs):
  config = os.environ.get('CONFIG', '').strip()
  step = int(os.environ.get('STEP', '0'))
  attrs.is_tuning = not config and step > 0
  tail_op, explicit_ops = None, [x for x in attrs.explicit_ops]

  if (len(explicit_ops) > 1 and
      not explicit_ops[-1].output(0).op.reduce_axis and
      len(explicit_ops[-1].output(0).op.input_tensors) <= 1):
    fuse_tail = attrs.auto_config.define_knob(f"FU", [False, True])
    if fuse_tail:
      tail_op, explicit_ops = explicit_ops[-1], explicit_ops[:-1]

  for rank, op in enumerate(reversed(explicit_ops)):
    _schedule_single(attrs, op.output(0), rank, tail_op is not None and rank == 0)
