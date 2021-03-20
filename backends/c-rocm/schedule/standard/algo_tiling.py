# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import os

def schedule_branch(attrs, output, prefix):
    cfg, s = attrs.auto_config, attrs.scheduler
    data_list, reduce_list = list(s[output].op.axis), list(s[output].op.reduce_axis)

    num_elements = 1
    for i, ax in enumerate(data_list):
      num_elements *= attrs.get_extent(ax)
      cfg.define_split(f"{prefix}D{i}", ax, num_outputs=4)
    for i, ax in enumerate(reduce_list):
      cfg.define_split(f"{prefix}R{i}", ax, num_outputs=3)

    input_list = []
    for I in s[output].op.input_tensors:
      input_list.append(I)

    num_threads, num_vthreads = 1, 1
    for i in range(len(data_list)):
      num_threads *= cfg[f"{prefix}D{i}"].size[2]
      num_vthreads *= cfg[f"{prefix}D{i}"].size[1] * cfg[f"{prefix}D{i}"].size[3]

    config = os.environ.get('CONFIG', '').strip()
    step = int(os.environ.get('STEP', '0'))
    if not config and step > 0:
      assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
      assert num_vthreads >= min(num_elements, 64), "Unrecommended small vthread counts: %d" % num_vthreads

    assert num_threads <= attrs.device_props.max_threads_per_block, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, attrs.device_props.max_threads_per_block)

    output, OL = s.cache_local(output)

    data_slices, reduce_slices = [], []
    for i in range(len(data_list)):
      data_slices.append(list(cfg[f"{prefix}D{i}"].apply(s, output, data_list[i])))

    for i in range(len(reduce_list)):
      reduce_slices.append(list(cfg[f"{prefix}R{i}"].apply(s, OL, reduce_list[i])))

    first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]
    s[output].reorder(*(first + second + third + fourth))
    s[OL].compute_at(s[output], third[-1])

    b_fused = s[output].fuse(*first)
    v_fused = s[output].fuse(*second)
    t_fused = s[output].fuse(*third)

    s[output].bind(b_fused, te.thread_axis("blockIdx.x"))
    s[output].bind(v_fused, te.thread_axis("vthread"))
    s[output].bind(t_fused, te.thread_axis("threadIdx.y"))

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
      cfg.define_knob(f"{prefix}V{i}", [1, 2, 4])
      fused_o, fused_i = s[load].split(fused_o, factor=cfg[f"{prefix}V{i}"].val)
      s[load].vectorize(fused_i)
      fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
      s[load].bind(fused_i, te.thread_axis("threadIdx.y"))

    # unroll
    cfg.define_knob(f"{prefix}S", [1, 4, 16, 64, 512])
    cfg.define_knob(f"{prefix}R", [False, True])
    kernel_scope = cache_loc
    s[OL].pragma(kernel_scope, 'auto_unroll_max_step', cfg[f"{prefix}S"].val)
    s[OL].pragma(kernel_scope, 'unroll_explicit', cfg[f"{prefix}R"].val)

