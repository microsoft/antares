# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import numpy as np

def schedule(antares):
  cfg, s, output = antares.auto_config, antares.scheduler, antares.outputs[0]
  th_vals, rd_vals = [antares.get_extent(x) for x in output.op.axis], [antares.get_extent(x) for x in output.op.reduce_axis]

  inputs = antares.inputs
  program = antares.ir

  import os
  plan_threads = 1

  def scpu_auto_schedule(s, output):
    cfg.define_knob("fuse_axis", [False])

    cfg.define_knob("pa_axis", np.arange(len(th_vals)).tolist())
    pa_id = cfg['pa_axis'].val

    ax_high, ax_low = [], []
    for i in range(len(th_vals)):
      ax = output.op.axis[i]
      cfg.define_split('axis_%d' % i, cfg.axis(ax), num_outputs=2)
      axm, axi = cfg['axis_%d' % i].apply(s, output, ax)

      if pa_id == i:
        axo, axm = s[output].split(axm, nparts=plan_threads)
        s[output].bind(axo, te.thread_axis('threadIdx.x'))

      ax_high.append(axm)
      ax_low.append(axi)

    for i in range(len(ax_high)):
      s[output].bind(ax_high[i], te.thread_axis('vthread'))
      s[output].bind(ax_low[i], te.thread_axis('vthread'))

    cfg.define_reorder("reorder", ax_low, "all")
    perm = cfg['reorder'].perm
    ex_ord = []
    for i in perm:
      ex_ord.append(ax_high[i])
    for i in perm:
      ex_ord.append(ax_low[i])
    s[output].reorder(*reversed(ex_ord))
    return

  return scpu_auto_schedule(s, output)

