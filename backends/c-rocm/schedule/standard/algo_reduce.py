# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix, tail_op):
  cfg, s = attrs.auto_config, attrs.scheduler

  rax = cfg.define_knob(f"{prefix}RA", [x for x in range(len(s[output].op.reduce_axis))])
  for i, ax in enumerate(s[output].op.reduce_axis):
    sizes = cfg.define_split(f"{prefix}R{i}", 1024 * 16, num_outputs=3, init_vals=[[-1, 1, attrs.device_props.warp_size]])
    if rax == i:
      use_wrap_reduce = cfg.define_knob(f"{prefix}U", [True, False])
      r_range = attrs.device_props.warp_size if use_wrap_reduce else sizes[-1]
      r_range = max(r_range, 2)
      ko, ki = s[output].split(ax, factor=r_range)
      s[output].bind(ki, te.thread_axis("threadIdx.x"))

  if tail_op is not None:
    output_ax = set([str(x.var.name) for x in s[output].op.axis])
    tail_ax = [i for i, x in enumerate(tail_op.output(0).op.axis) if str(x.var.name) not in output_ax]
    if not tail_ax:
      tax = -1
    else:
      tax = cfg.define_knob(f"{prefix}T", tail_ax)
  else:
    tax = -1

  data_slices, main_op = [], tail_op or output
  for i, ax in enumerate(s[main_op].op.axis):
    sizes = cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=4)
    data_slices.append([-1, sizes[1] * sizes[2] * sizes[3]])

  if not getattr(tail_op, 'is_fused', False):
    data_slices = data_slices[:len(s[output].op.axis)]
    for i, ax in enumerate(s[output].op.axis):
      data_slices[i] = list(cfg.apply_split(s, output, ax, data_slices[i]))
    first, second = [x[0] for x in data_slices], [x[1] for x in data_slices]
    s[output].reorder(*(first + second))
    fused_b = s[output].fuse(*first)
    fused_t = s[output].fuse(*second)
    s[output].bind(fused_b, te.thread_axis("blockIdx.x"))
    s[output].bind(fused_t, te.thread_axis("threadIdx.y"))
    return

  for i, ax in enumerate(s[tail_op].op.axis):
    if tax != i:
      data_slices[i] = list(cfg.apply_split(s, tail_op, ax, data_slices[i]))
    else:
      data_slices[i] = list(s[tail_op].split(ax, factor=r_range))
      s[tail_op].bind(data_slices[i][-1], te.thread_axis("threadIdx.x"))

  first, second = [x[0] for i, x in enumerate(data_slices) if i != tax], [x[1] for i, x in enumerate(data_slices) if i != tax]
  s[tail_op].reorder(*(first + second))
  outer_ax = s[tail_op].fuse(*first)
  inner_ax = s[tail_op].fuse(*second)
  s[tail_op].bind(outer_ax, te.thread_axis("blockIdx.x"))
  s[tail_op].bind(inner_ax, te.thread_axis("threadIdx.y"))
  s[output].compute_at(s[tail_op], inner_ax)

