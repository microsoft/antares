# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tvm
import numpy as np


def schedule(antares):
  cfg, s, output = antares.auto_config, antares.scheduler, antares.outputs[0]
  th_vals, rd_vals = [antares.get_extent(x) for x in output.op.axis], [antares.get_extent(x) for x in output.op.reduce_axis]

  inputs = antares.inputs
  program = antares.ir

  import os
  plan_threads = int(os.environ.get('CPU_THREADS', '8'))

  def mcpu_auto_schedule(s, output):
    cfg.define_knob("fuse_axis", [False])
    # fused = s[output].fuse(.. output.op.axis ..)
    # if len(rd_vals) > 0:
    #   if output.op in s.outputs:
    #     output_local = s.cache_write(output, "local")
    #   else:
    #     s[output].set_scope('local')
    #     output_local, output = output, s.outputs[0].output(0)

    cfg.define_knob("pa_axis", np.arange(len(th_vals)).tolist())
    pa_id = cfg['pa_axis'].val

    ax_high, ax_low = [], []
    for i in range(len(th_vals)):
      ax = output.op.axis[i]
      cfg.define_split('axis_%d' % i, cfg.axis(ax), num_outputs=2)
      axm, axi = cfg['axis_%d' % i].apply(s, output, ax)

      if pa_id == i:
        axo, axm = s[output].split(axm, nparts=plan_threads)
        s[output].bind(axo, tvm.thread_axis('threadIdx.x'))

      ax_high.append(axm)
      ax_low.append(axi)

    for i in range(len(ax_high)):
      s[output].bind(ax_high[i], tvm.thread_axis('vthread'))
      s[output].bind(ax_low[i], tvm.thread_axis('vthread'))

    cfg.define_reorder("reorder", ax_low, "all")
    perm = cfg['reorder'].perm
    ex_ord = []
    for i in perm:
      ex_ord.append(ax_high[i])
    for i in perm:
      ex_ord.append(ax_low[i])
    s[output].reorder(*reversed(ex_ord))

    # if len(rd_vals) > 0:
    #   s[output_local].compute_at(s[output], ex_ord[0])
    return

  return mcpu_auto_schedule(s, output)

