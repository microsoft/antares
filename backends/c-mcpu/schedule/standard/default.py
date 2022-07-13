# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
from antares.common import product

def schedule(attrs):
  cfg, s = attrs.auto_config, attrs.scheduler

  def mcpu_auto_schedule(s, output, prefix):
    hyper_params = [[-1, 2, 8, 4], [-1, 1, 512, 1]]
    slice_data, slice_reduce = [], []
    for i in range(len(output.op.axis)):
      slice_data.append(cfg.define_split(f"{prefix}:D{i}", attrs.get_extent(output.op.axis[i]), num_outputs=4, init_vals=[hyper_params[i % len(hyper_params)],]))
    for i in range(len(output.op.reduce_axis)):
      slice_reduce.append(cfg.define_split(f"{prefix}:R{i}", attrs.get_extent(output.op.reduce_axis[i]), num_outputs=2, init_vals=[[-1, 4],]))

    unroll = cfg.define_knob(f"{prefix}:UN", [1, 4, 8, 16, 32, 64], init_vals=[1,] if attrs.backend == 'c-mcpu_avx512' else [0,])

    output_local, = s.cache_write([output], "local")

    slice_axes = []
    for i in range(len(output.op.axis)):
      slice_axes.append(cfg.apply_split(s, output_local, output_local.op.axis[i], slice_data[i]))

    if output.op.reduce_axis:
      reduce_at = cfg.define_knob(f"{prefix}:RA", [x for x in range(len(output.op.reduce_axis))], init_vals=[0,])
      output_local_K_o, output_local_K_i = cfg.apply_split(s, output_local, output_local.op.reduce_axis[reduce_at], slice_reduce[reduce_at])
      output_local_K_o, output_local_K_i = [output_local_K_o], [output_local_K_i]
    else:
      output_local_K_o, output_local_K_i = [], []

    first, second, third, fourth = [x[0] for x in slice_axes], [x[1] for x in slice_axes], [x[2] for x in slice_axes], [x[3] for x in slice_axes]
    s[output_local].reorder(*(first + second + output_local_K_o + third + output_local_K_i + fourth))

    slice_global_axes = []
    for i in range(len(output.op.axis)):
      if cfg.define_knob(f"{prefix}:_{i}", [False, True], init_vals=[0,]):
        slice_global_axes.append(cfg.apply_split(s, output, output.op.axis[i], [-1, slice_data[i][1], int(product(slice_data[i][2:]))]))
      else:
        slice_global_axes.append(cfg.apply_split(s, output, output.op.axis[i], [-1, 1, int(product(slice_data[i][1:]))]))

    s[output].reorder(*([x[0] for x in slice_global_axes] + [x[1] for x in slice_global_axes] + [x[2] for x in slice_global_axes]))

    s[output_local].compute_at(s[output], slice_global_axes[-1][1])
    s[output].bind(s[output].fuse(*[x[0] for x in slice_global_axes]), te.thread_axis('threadIdx.x'))

    s[output_local].pragma(first[0], "auto_unroll_max_step", unroll)
    s[output_local].pragma(first[0], "unroll_explicit", True)
    # s[output_local].vectorize(fourth[-1])
    s[output_local].unroll(fourth[-1])

  def mcpu_simple_schedule(s, output, prefix):
    slice_data = [cfg.define_split(f"{prefix}:D{i}", attrs.get_extent(output.op.axis[i]), num_outputs=3, init_vals=[[-1, 1, 1],]) for i in range(len(output.op.axis))]
    slice_axes = [cfg.apply_split(s, output, output.op.axis[i], [-1, 1] + slice_data[i][1:]) for i in range(len(output.op.axis))]

    first, second, third, fourth = [x[0] for x in slice_axes], [x[1] for x in slice_axes], [x[2] for x in slice_axes], [x[3] for x in slice_axes]
    s[output].reorder(*(first + second + third + fourth))

    s[output].bind(s[output].fuse(*first), te.thread_axis('threadIdx.x'))
    s[output].bind(s[output].fuse(*second), te.thread_axis('vthread'))


  for i, m in enumerate(attrs.explicit_ops):
    if len(m.output(0).op.reduce_axis) == 0:
      return mcpu_simple_schedule(s, m.output(0), f'T{m.output(0).name}')
    mcpu_auto_schedule(s, m.output(0), f'T{m.output(0).name}')

