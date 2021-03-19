# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess


def schedule(attrs):
    cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.explicit_ops[-1].output(0)
    data_list, reduce_list = list(s[output].op.axis), list(s[output].op.reduce_axis)

    for i, ax in enumerate(data_list):
      cfg.define_split(f"D{i}", ax, num_outputs=4)
    for i, ax in enumerate(reduce_list):
      cfg.define_split(f"R{i}", ax, num_outputs=3)

    input_list = []
    for I in s[output].op.input_tensors:
      input_list.append(I)

    num_threads = 1
    for i in range(len(data_list)):
      num_threads *= cfg[f"D{i}"].size[2]
    assert num_threads <= 1024, "Invalid schedule plans: num_threads(%d) > 1024" % num_threads

    if output.op in s.outputs:
      output = output
      OL = s.cache_write(output, 'local')
    else:
      output = s.outputs[0].output(0)
      s[output].set_scope('local')
      OL = output

    data_slices, reduce_slices = [], []
    for i in range(len(data_list)):
      data_slices.append(list(cfg[f"D{i}"].apply(s, output, data_list[i])))

    for i in range(len(reduce_list)):
      reduce_slices.append(list(cfg[f"R{i}"].apply(s, OL, reduce_list[i])))

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
      cfg.define_knob(f"V{i}", [1, 2, 4])
      fused_o, fused_i = s[load].split(fused_o, factor=cfg[f"V{i}"].val)
      s[load].vectorize(fused_i)
      fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
      s[load].bind(fused_i, te.thread_axis("threadIdx.y"))

    # unroll
    cfg.define_knob("auto_unroll_max_step", [1, 4, 16, 32, 64, 512, 1024])
    cfg.define_knob("unroll_explicit", [False, True])
    kernel_scope = cache_loc
    s[OL].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[OL].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

