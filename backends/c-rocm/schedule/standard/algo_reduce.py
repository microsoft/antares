# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler

  use_wrap_reduce = cfg.define_knob(f"{prefix}W", [False, True])
  rax = cfg.define_knob(f"{prefix}S", [x for x in range(len(s[output].op.reduce_axis))])
  for i, ax in enumerate(s[output].op.reduce_axis):
    sizes = cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=2)
    if rax == i:
      r_range = attrs.device_props.warp_size if use_wrap_reduce else sizes[1]
      r_range = max(r_range, 2)
      ko, ki = s[output].split(ax, factor=r_range)
      BF = s.rfactor(output, ki)

  data_slices = []
  for i, ax in enumerate(s[output].op.axis):
    sizes = cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=2)
    data_slices.append(list(cfg.apply_split(s, output, ax, sizes)))

  first, second = [x[0] for x in data_slices], [x[1] for x in data_slices]
  s[output].reorder(*(first + second))

  fused_bx = s[output].fuse(*first)
  fused_by = s[output].fuse(*second)

  s[output].bind(fused_bx, te.thread_axis("blockIdx.x"))
  s[output].bind(fused_by, te.thread_axis("blockIdx.y"))

  rax = min(rax, len(s[output].op.reduce_axis) - 1)
  reduce_ax = s[output].op.reduce_axis[rax]
  tx = te.thread_axis("threadIdx.x")
  s[output].bind(reduce_ax, tx)
  s[BF].compute_at(s[output], reduce_ax)
  s[output].set_store_predicate(tx.var.equal(0))
