# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler

  rax = cfg.define_knob(f"{prefix}S", [x for x in range(len(s[output].op.reduce_axis))])
  for i, ax in enumerate(s[output].op.reduce_axis):
    sizes = cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=2)
    if rax == i:
      use_wrap_reduce = cfg.define_knob(f"{prefix}W", [False, True])
      r_range = attrs.device_props.warp_size if use_wrap_reduce else sizes[1]
      r_range = max(r_range, 2)
      ko, ki = s[output].split(ax, factor=r_range)
      s[output].bind(ki, te.thread_axis("threadIdx.x"))

  data_slices = []
  for i, ax in enumerate(s[output].op.axis):
    sizes = cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=2)
    data_slices.append(list(cfg.apply_split(s, output, ax, sizes)))

  first, second = [x[0] for x in data_slices], [x[1] for x in data_slices]
  s[output].reorder(*(first + second))
  s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
  s[output].bind(s[output].fuse(*second), te.thread_axis("blockIdx.y"))
