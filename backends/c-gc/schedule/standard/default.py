# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tvm import te


def schedule(attrs):
  cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
  th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

  inputs = attrs.inputs
  program = attrs.ir

  # Global tuning space
  if not os.environ.get('CONFIG', '') and int(os.environ.get('STEP', '0')) > 0:
    for i in range(len(output.op.axis)):
      ax_name = 'axis_%d' % i
      cfg.define_split(ax_name, cfg.axis(output.op.axis[i]), num_outputs=2)
    # num_cores, align_width = 1216, 64
    # cfg.define_knob('start_core', [x * align_width for x in range(num_cores // align_width)])
    return

  loop_axes = []
  for i in range(len(th_vals)):
    lo, li = s[output].split(output.op.axis[i], nparts=1)
    if i == 0:
      s[output].bind(lo, te.thread_axis('blockIdx.x'))
    else:
      li = s[output].fuse(lo, li)
    s[output].bind(li, te.thread_axis('vthread'))
    loop_axes.append(li)

  s[output].reorder(*reversed(loop_axes))
