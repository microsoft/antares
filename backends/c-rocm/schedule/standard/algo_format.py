# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler
  th_vals = [attrs.get_extent(x) for x in output.op.axis]

  # Normal Schedule Plan
  blocks = [te.thread_axis('blockIdx.x'), te.thread_axis('blockIdx.y'), te.thread_axis('blockIdx.z')]
  threads = [te.thread_axis('threadIdx.x'), te.thread_axis('threadIdx.y'), te.thread_axis('threadIdx.z')]

  th_idx = []
  for i in range(len(th_vals)):
    if th_vals[i] > 1 or (i + 1 == len(th_vals) and len(th_idx) == 0):
      th_idx.append(i)
    else:
      s[output].bind(output.op.axis[i], te.thread_axis('vthread'))

  high_vaxis, low_vaxis = [], []
  for i in range(len(th_idx)):
    ax_name = f'{prefix}D{th_idx[i]}'
    ax_obj = output.op.axis[th_idx[i]]
    if i < len(blocks):
      sizes = cfg.define_split(ax_name, attrs.get_extent(ax_obj), num_outputs=4)
      ax1, ax2, ax3, ax4 = cfg.apply_split(s, output, ax_obj, sizes)
      s[output].bind(ax1, blocks[i])
      s[output].bind(ax3, threads[i])
    else:
      sizes = cfg.define_split(ax_name, attrs.get_extent(ax_obj), num_outputs=2)
      ax2, ax4 = cfg.apply_split(s, output, ax_obj, sizes)
    s[output].bind(ax2, te.thread_axis('vthread'))
    s[output].bind(ax4, te.thread_axis('vthread'))
    high_vaxis.append(ax2)
    low_vaxis.append(ax4)

  ord_name = f"{prefix}O"
  permut = cfg.define_reorder(ord_name, len(high_vaxis), "all")
  plan_order = []
  for i in permut:
    plan_order.append(low_vaxis[i])
    plan_order.append(high_vaxis[i])
  s[output].reorder(*plan_order)

  # unroll
  unroll_step = cfg.define_knob(f"{prefix}S", [1, 4, 16, 64, 512])
  unroll_explicit = cfg.define_knob(f"{prefix}R", [False, True])
  kernel_scope = plan_order[0]
  s[output].pragma(kernel_scope, 'auto_unroll_max_step', unroll_step)
  s[output].pragma(kernel_scope, 'unroll_explicit', unroll_explicit)
