# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tvm


def schedule(antares):
  cfg, s, output = antares.auto_config, antares.scheduler, antares.outputs[0]
  th_vals, rd_vals = [antares.get_extent(x) for x in output.op.axis], [antares.get_extent(x) for x in output.op.reduce_axis]

  inputs = antares.inputs
  program = antares.ir

  # Global tuning space
  if not os.environ.get('CONFIG', '') and int(os.environ.get('STEP', '0')) > 0:
    for i in range(len(output.op.axis)):
      ax_name = 'axis_%d' % i
      cfg.define_split(ax_name, cfg.axis(output.op.axis[i]), num_outputs=2)
    num_cores, align_width = 1216, 64
    cfg.define_knob('start_core', [x * align_width for x in range(num_cores // align_width)])
    return

  # Local scheduling plan
  if rd_vals:
    if output.op in s.outputs:
      output_local = s.cache_write(output, "local")
    else:
      s[output].set_scope('local')
      output_local, output = output, s.outputs[0].output(0)

  loop_axes = []
  for i in range(len(th_vals)):
    lo, li = s[output].split(output.op.axis[i], nparts=1)
    if i == 0:
      s[output].bind(lo, tvm.thread_axis('threadIdx.x'))
    else:
      li = s[output].fuse(lo, li)
    s[output].bind(li, tvm.thread_axis('vthread'))
    loop_axes.append(li)

  s[output].reorder(*(loop_axes))

  if rd_vals:
    s[output_local].compute_at(s[output], loop_axes[-1])
