# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix, tail_op):
  cfg, s = attrs.auto_config, attrs.scheduler

  rax = cfg.define_knob(f"{prefix}S", [x for x in range(len(s[output].op.reduce_axis))])
  for i, ax in enumerate(s[output].op.reduce_axis):
    sizes = cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=2, init_vals=[[-1, attrs.device_props.warp_size]])
    if rax == i:
      use_wrap_reduce = cfg.define_knob(f"{prefix}W", [True, False])
      r_range = attrs.device_props.warp_size if use_wrap_reduce else sizes[1]
      r_range = max(r_range, 2)
      ko, ki = s[output].split(ax, factor=r_range)
      s[output].bind(ki, te.thread_axis("threadIdx.x"))

  if len(attrs.explicit_ops[-1].output(0).op.reduce_axis) == 0:
    length = attrs.get_extent(s[output].op.reduce_axis[rax])
    match_list = [i for i, ax in enumerate(attrs.explicit_ops[-1].output(0).op.axis) if attrs.get_extent(ax) == length]
    tax = cfg.define_knob(f"{prefix}T", match_list) if len(match_list) > 0 else -1

  if not tail_op:
    s[output].reorder(*s[output].op.axis)
    outer_ax = s[output].fuse(*s[output].op.axis)
    s[output].bind(outer_ax, te.thread_axis("blockIdx.x"))
    return

  axes, extra = s[tail_op].op.axis[:tax] + s[tail_op].op.axis[tax+1:], s[tail_op].op.axis[tax]
  s[tail_op].reorder(*axes)
  outer_ax = s[tail_op].fuse(*axes)
  s[tail_op].bind(outer_ax, te.thread_axis("blockIdx.x"))

  outer_o, outer_i = s[tail_op].split(extra, factor=r_range)
  s[tail_op].bind(outer_i, te.thread_axis("threadIdx.x"))
  s[output].compute_at(s[tail_op], outer_ax)
