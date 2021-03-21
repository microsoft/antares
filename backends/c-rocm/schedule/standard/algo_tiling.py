# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import os

def schedule_branch(attrs, output, prefix):
    cfg, s = attrs.auto_config, attrs.scheduler

    data_sizes, reduce_sizes = [], []
    num_elements = 1
    for i, ax in enumerate(s[output].op.axis):
      num_elements *= attrs.get_extent(ax)
      data_sizes.append(cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=4))
    for i, ax in enumerate(s[output].op.reduce_axis):
      reduce_sizes.append(cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=3))

    num_threads, num_vthreads = 1, 1
    for i in range(len(s[output].op.axis)):
      num_threads *= data_sizes[i][2]
      num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

    config = os.environ.get('CONFIG', '').strip()
    step = int(os.environ.get('STEP', '0'))
    if not config and step > 0:
      assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
      # assert num_vthreads >= min(num_elements, 64), "Unrecommended small vthread counts: %d" % num_vthreads

    assert num_threads <= attrs.device_props.max_threads_per_block, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, attrs.device_props.max_threads_per_block)

    input_list = []
    for I in s[output].op.input_tensors:
      input_list.append(I)

    output, OL = s.cache_local(output)

    data_list, reduce_list = list(s[output].op.axis), list(s[OL].op.reduce_axis)

    data_slices, reduce_slices = [], []
    for i in range(len(data_list)):
      data_slices.append(list(cfg.apply_split(s, output, data_list[i], data_sizes[i])))

    for i in range(len(reduce_list)):
      reduce_slices.append(list(cfg.apply_split(s, OL, reduce_list[i], reduce_sizes[i])))

    first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]
    s[output].reorder(*(first + second + third + fourth))
    s[OL].compute_at(s[output], third[-1])

    b_fused = s[output].fuse(*first)
    v_fused = s[output].fuse(*second)
    t_fused = s[output].fuse(*third)

    s[output].bind(b_fused, te.thread_axis("blockIdx.x"))
    s[output].bind(v_fused, te.thread_axis("vthread"))
    s[output].bind(t_fused, te.thread_axis("threadIdx.x"))

    for i in range(len(input_list)):
      input_list[i] = s.cache_read(input_list[i], 'shared', [OL])

    # tile reduction axes
    reduce_order = [x[0] for x in reduce_slices] + [x[1] for x in reduce_slices] + [x[2] for x in reduce_slices]
    s[OL].reorder(*(reduce_order + list(s[OL].op.axis)))
    cache_loc = reduce_order[0]

    for IS in input_list:
      s[IS].compute_at(s[OL], cache_loc)

    # cooperative fetching
    for i, load in enumerate(input_list):
      fused_o = s[load].fuse(*s[load].op.axis)
      val = cfg.define_knob(f"{prefix}V{i}", [1, 2, 4])
      fused_o, fused_i = s[load].split(fused_o, factor=val)
      s[load].vectorize(fused_i)
      fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
      s[load].bind(fused_i, te.thread_axis("threadIdx.x"))

    # unroll
    unroll_step = cfg.define_knob(f"{prefix}S", [1, 4, 16, 64, 512])
    unroll_explicit = cfg.define_knob(f"{prefix}R", [False, True])
    kernel_scope = cache_loc
    s[OL].pragma(kernel_scope, 'auto_unroll_max_step', unroll_step)
    s[OL].pragma(kernel_scope, 'unroll_explicit', unroll_explicit)

