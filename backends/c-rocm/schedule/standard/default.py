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
      output = s.outputs[0].output(0)
      s[output].set_scope('local')
      OL = output
    return output, OL
  s.cache_local = cache_local
  s.have_tail = have_tail

  num_inputs = len(s[output].op.input_tensors)

  # Rough classification of computing features
  if num_inputs > 1 and len(output.op.reduce_axis) > 0:
    from .algo_tiling import schedule_branch
    return schedule_branch(attrs, output, f"T{rank}:")

  from .algo_format import schedule_branch
  return schedule_branch(attrs, output, f"F{rank}:")

def schedule(attrs):
  have_tail, explicit_ops = False, [x for x in attrs.explicit_ops]
  if len(explicit_ops) > 1 and not explicit_ops[-1].output(0).op.reduce_axis:
    have_tail = True
    explicit_ops = explicit_ops[:-1]
  for rank, op in enumerate(explicit_ops):
    _schedule_single(attrs, op.output(0), rank, have_tail and rank + 1 == len(explicit_ops))
