# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import numpy as np

def schedule(attrs):
  cfg, s = attrs.auto_config, attrs.scheduler

  if attrs.backend == 'c-scpu':
    plan_threads = 1
  else:
    import os, multiprocessing
    plan_threads = os.environ.get('CPU_THREADS', '')
    if not plan_threads:
      plan_threads = str(multiprocessing.cpu_count())
    plan_threads = int(plan_threads)

  def mcpu_auto_schedule(s, output, rank):
    axo, axm = s[output].split(output.op.axis[0], nparts=plan_threads)
    s[output].bind(axo, te.thread_axis('threadIdx.x'))
    s[output].bind(axm, te.thread_axis('vthread'))
    for ax in output.op.axis[1:]:
      s[output].bind(ax, te.thread_axis('vthread'))
    return

  for i, m in enumerate(attrs.explicit_ops):
    mcpu_auto_schedule(s, m.output(0), i)
