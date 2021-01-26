# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess
import json
import os


def schedule(attrs):
  cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
  th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

  cfg.define_reorder("reorder_axis", [i for i in range(len(output.op.axis))], "all")
  perm = cfg["reorder_axis"].perm

  vthreads = []
  for i in range(len(th_vals)):
    if i < 3:
      s[output].bind(output.op.axis[perm[i]], te.thread_axis(f'blockIdx.{i}'))
    else:
      s[output].bind(output.op.axis[perm[i]], te.thread_axis(f'vthread'))
      vthreads.append(output.op.axis[perm[i]])

  s[output].reorder(*vthreads)
