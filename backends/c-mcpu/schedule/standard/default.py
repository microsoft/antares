# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import numpy as np

def schedule(attrs):
  cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
  th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

  inputs = attrs.inputs
  program = attrs.ir

  if attrs.backend == 'c-scpu':
    plan_threads = 1
  else:
    import os, multiprocessing
    plan_threads = os.environ.get('CPU_THREADS', '')
    if not plan_threads:
      plan_threads = str(multiprocessing.cpu_count())
    plan_threads = int(plan_threads)

  def mcpu_auto_schedule(s, output):
    cfg.define_knob("fuse_axis", [False])
    # fused = s[output].fuse(.. output.op.axis ..)

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

    for m in attrs.explicit_ops[:-1]:
      s[m.output(0)].compute_at(s[output], ex_ord[0])
    return

  return mcpu_auto_schedule(s, output)

