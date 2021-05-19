# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from tvm import te

def schedule(attrs):
  cfg, s = attrs.auto_config, attrs.scheduler
  assert len(attrs.explicit_ops) == 1, "Unhandled multiple explicit-op scheduling."
  output = attrs.explicit_ops[0].output(0)
  program = attrs.ir

  # Global tuning space
  if not os.environ.get('CONFIG', '') and int(os.environ.get('STEP', '0')) > 0:
    for i in range(len(output.op.axis)):
      cfg.define_split('tile_%d' % i, attrs.get_extent(output.op.axis[i]), num_outputs=3)
    # num_cores, align_width = 1216, 64
    # cfg.define_knob('start_core', [x * align_width for x in range(num_cores // align_width)])
    return

  configs = os.environ.get('CONFIG', '').strip()
  configs = json.loads(configs) if configs else {}
  inner_tiles = [configs.get(f'tile_{i}', [-1, 1, 1])[-1] for i in range(len(output.op.axis))]

  output_local = s.cache_write(output, 'local')

  data_slices = []
  for i in range(len(output.op.axis)):
    assert attrs.get_extent(output.op.axis[i]) % inner_tiles[i] == 0, f"Outer tiling & inner tiling mismatch: {attrs.get_extent(output.op.axis[i])} v.s. {inner_tiles[i]}"
    axo, axm, axi = cfg.apply_split(s, output_local, output_local.op.axis[i], [-1, attrs.get_extent(output.op.axis[i]) // inner_tiles[i], inner_tiles[i]])
    data_slices.append((axm, axi))

  s[output_local].reorder(*([x[0] for x in data_slices] + [x for x in output_local.op.reduce_axis] + [x[1] for x in data_slices]))

  data_slices = [list(cfg.apply_split(s, output, output.op.axis[i], [-1, attrs.get_extent(output.op.axis[i]) // inner_tiles[i], inner_tiles[i]])) for i in range(len(output.op.axis))]
  s[output].reorder(*([x[0] for x in data_slices] + [x[1] for x in data_slices] + [x[2] for x in data_slices]))
  s[output_local].compute_at(s[output], data_slices[-1][1])

  s[output].bind(s[output].fuse(*[x[0] for x in data_slices]), te.thread_axis('blockIdx.x'))
