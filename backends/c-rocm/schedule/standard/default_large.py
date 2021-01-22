# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess

import json
import os
import itertools

def schedule(attrs):
  cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
  th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]
  data_axis = output.op.axis

  if rd_vals:
    if output.op in s.outputs:
      output_local = s.cache_write(output, "local")
    else:
      s[output].set_scope('local')
      output_local, output = output, s.outputs[0].output(0)

  fuse_space = []
  for npart in range(min(4, len(th_vals))):
    fuse_space.extend(itertools.combinations(range(1, len(th_vals)), npart))

  cfg.define_knob('fuse', range(len(fuse_space)))
  fuse_inst = [0] + list(fuse_space[cfg['fuse'].val]) + [len(th_vals)]

  fused_axis, fused_vals, data_region = [], [], []
  for i in range(1, len(fuse_inst)):
    data_region += [len(fused_axis)] * (fuse_inst[i] - fuse_inst[i - 1])
    fused_axis.append(s[output].fuse(*data_axis[fuse_inst[i-1]:fuse_inst[i]]))
    fused_vals.append(int(np.product(th_vals[fuse_inst[i-1]:fuse_inst[i]])))

  mapping_space = [
    ['blockIdx.x', 'threadIdx.x'],
    ['blockIdx.y', 'threadIdx.y'],
    ['blockIdx.z', 'vthread' if len(fused_axis) > 3 else 'threadIdx.z'],
    ['threadIdx.z', 'vthread'],
  ]
  cfg.define_reorder('maps', [x for x in range(len(mapping_space))], 'all')

  fused_splits = [[-1, 1, 1, 1]] * len(fused_axis)
  for i in range(len(output.op.axis)):
    ax_name = 'axis_%d' % i
    cfg.define_split(ax_name, th_vals[i], num_outputs=4)
    for rank, factor in enumerate(cfg[ax_name].size):
      if rank >= 1:
        fused_splits[data_region[i]][rank] *= factor

  reorder_list = []
  for rank, bx, val, splits in zip(range(len(fused_axis)), fused_axis, fused_vals, fused_splits):
    bx, xi = s[output].split(bx, factor=splits[3])
    bx, tx = s[output].split(bx, factor=splits[2])
    bx, xo = s[output].split(bx, factor=splits[1])

    target_threads = mapping_space[cfg['maps'].perm[rank]]
    s[output].bind(bx, te.thread_axis(target_threads[0]))
    if target_threads[1] == 'vthread':
      xi = s[output].fuse(xo, tx, xi)
    else:
      s[output].bind(xo, te.thread_axis('vthread'))
      reorder_list.append(xo)
      s[output].bind(tx, te.thread_axis(target_threads[1]))
    s[output].bind(xi, te.thread_axis('vthread'))
    reorder_list.append(xi)

  s[output].reorder(*reorder_list)
  if rd_vals:
    s[output_local].compute_at(s[output], reorder_list[0])
    for i in range(len(rd_vals)):
      if rd_vals[i] > 1:
        ax_name = 'reduce_%d' % i
        cfg.define_split(ax_name, cfg.axis(output_local.op.reduce_axis[i]), num_outputs=3)
        ko, kt, ki = cfg[ax_name].apply(s, output_local, output_local.op.reduce_axis[i])
        s[output_local].unroll(kt)
