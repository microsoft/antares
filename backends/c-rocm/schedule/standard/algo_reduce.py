# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler

  rax = cfg.define_knob(f"{prefix}S", [x for x in range(len(s[output].op.reduce_axis))])
  for i, ax in enumerate(s[output].op.reduce_axis):
    sizes = cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=2)
    if rax == i:
      r_range = max(2, sizes[1])
      ko, ki = s[output].split(ax, factor=r_range)
      BF = s.rfactor(output, ki)

  data_slices = []
  for i, ax in enumerate(s[output].op.axis):
    sizes = cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=2)
    data_slices.append(list(cfg.apply_split(s, output, ax, sizes)))

  first, second = [x[0] for x in data_slices], [x[1] for x in data_slices]
  s[output].reorder(*(first + second))

  fused_b = s[output].fuse(*first)
  fused_t = s[output].fuse(*second)

  s[output].bind(fused_b, te.thread_axis("blockIdx.x"))
  s[output].bind(fused_t, te.thread_axis("threadIdx.y"))

  rax = min(rax, len(s[output].op.reduce_axis) - 1)
  reduce_ax = s[output].op.reduce_axis[rax]
  tx = te.thread_axis("threadIdx.x")
  s[output].bind(reduce_ax, tx)
  s[BF].compute_at(s[output], reduce_ax)
  s[output].set_store_predicate(tx.var.equal(0))
