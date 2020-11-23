# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te as tvm
import logging
import sys, time, subprocess
import json
import os


def schedule(attrs):
  cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
  th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

  # Normal Schedule Plan
  blocks = [te.thread_axis('blockIdx.x'), te.thread_axis('blockIdx.y'), te.thread_axis('blockIdx.z')]
  threads = [te.thread_axis('threadIdx.x'), te.thread_axis('threadIdx.y'), te.thread_axis('threadIdx.z')]

  if rd_vals:
    if output.op in s.outputs:
      output_local = s.cache_write(output, "local")
    else:
      s[output].set_scope('local')
      output_local, output = output, s.outputs[0].output(0)

  th_idx = []
  for i in range(len(th_vals)):
    if th_vals[i] > 1 or (i + 1 == len(th_vals) and len(th_idx) == 0):
      th_idx.append(i)
    else:
      s[output].bind(output.op.axis[i], te.thread_axis('vthread'))

  high_vaxis, low_vaxis = [], []
  for i in range(len(th_idx)):
    ax_name = 'axis_%d' % th_idx[i]
    ax_obj = output.op.axis[th_idx[i]]
    if i < len(blocks):
      cfg.define_split(ax_name, cfg.axis(ax_obj), num_outputs=4)
      ax1, ax2, ax3, ax4 = cfg[ax_name].apply(s, output, ax_obj)
      s[output].bind(ax1, blocks[i])
      s[output].bind(ax3, threads[i])
    else:
      cfg.define_split(ax_name, cfg.axis(ax_obj), num_outputs=2)
      ax2, ax4 = cfg[ax_name].apply(s, output, ax_obj)
    s[output].bind(ax2, te.thread_axis('vthread'))
    s[output].bind(ax4, te.thread_axis('vthread'))
    high_vaxis.append(ax2)
    low_vaxis.append(ax4)

  cfg.define_reorder("reorder", high_vaxis, "all")
  plan_order = []
  for i in cfg["reorder"].perm:
    plan_order.append(low_vaxis[i])
    plan_order.append(high_vaxis[i])
  s[output].reorder(*plan_order)

  if rd_vals:
    s[output_local].compute_at(s[output], ax2)
    for i in range(len(rd_vals)):
      if rd_vals[i] > 1:
        ax_name = 'reduce_%d' % i
        cfg.define_split(ax_name, cfg.axis(output_local.op.reduce_axis[i]), num_outputs=3)
        ko, kt, ki = cfg[ax_name].apply(s, output_local, output_local.op.reduce_axis[i])
        s[output_local].unroll(kt)
