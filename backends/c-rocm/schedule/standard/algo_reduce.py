# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler

  cfg.define_knob(f"{prefix}S", [x for x in range(len(s[output].op.reduce_axis))])
  rax = cfg[f"{prefix}S"].val
  for i, ax in enumerate(s[output].op.reduce_axis):
    cfg.define_split(f"{prefix}R{i}", ax, num_outputs=2)
    if rax == i:
      r_range = max(2, cfg[f"{prefix}R{i}"].size[1])
      if not attrs.backend.startswith('c-cuda'):
        r_range = r_range if r_range != 32 else 16
      ko, ki = s[output].split(ax, factor=r_range)
      BF = s.rfactor(output, ki)

  data_list, reduce_list = list(s[output].op.axis), list(s[output].op.reduce_axis)
  for i, ax in enumerate(data_list):
    cfg.define_split(f"{prefix}D{i}", ax, num_outputs=2)

  data_slices = []
  for i, ax in enumerate(data_list):
    data_slices.append(list(cfg[f"{prefix}D{i}"].apply(s, output, ax)))

  first, second = [x[0] for x in data_slices], [x[1] for x in data_slices]
  s[output].reorder(*(first + second))

  fused_b = s[output].fuse(*first)
  fused_t = s[output].fuse(*second)

  s[output].bind(fused_b, te.thread_axis("blockIdx.x"))
  s[output].bind(fused_t, te.thread_axis("threadIdx.y"))

  tx = te.thread_axis("threadIdx.x")
  s[output].bind(reduce_list[rax], tx)
  s[BF].compute_at(s[output], reduce_list[rax])
  s[output].set_store_predicate(tx.var.equal(0))
