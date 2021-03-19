# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess


def schedule(attrs):
    cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.explicit_ops[-1].output(0)
    th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

    y, x = s[output].op.axis
    [rc] = s[output].op.reduce_axis
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)

    data_deform, kernel = s[output].op.input_tensors

    if output.op in s.outputs:
        output = output
        OL = s.cache_write(output, 'local')
    else:
        output = s.outputs[0].output(0)
        s[output].set_scope('local')
        OL = output

    # tile and bind spatial axes
    y, x = s[output].op.axis

    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    rco, rcm, rci = cfg['tile_rc'].apply(s, OL, rc)

    s[output].reorder(by, bx, vy, vx, ty, tx, yi, xi)
    s[OL].compute_at(s[output], tx)

    b_fused = s[output].fuse(by, bx)
    v_fused = s[output].fuse(vy, vx)
    t_fused = s[output].fuse(ty, tx)

    s[output].bind(b_fused, te.thread_axis("blockIdx.x"))
    s[output].bind(v_fused, te.thread_axis("vthread"))
    s[output].bind(t_fused, te.thread_axis("threadIdx.x"))

    # create cache stage
    AA = s.cache_read(data_deform, 'shared', [OL])
    WW = s.cache_read(kernel, 'shared', [OL])

    # tile reduction axes
    y, x = s[OL].op.axis
    [rc] = s[OL].op.reduce_axis
    s[OL].reorder(rco, rcm, rci, y, x)

    cache_loc = [rco][-1]
    s[AA].compute_at(s[OL], cache_loc)
    s[WW].compute_at(s[OL], cache_loc)

    # cooperative fetching
    for i, load in enumerate([AA, WW]):
        fused_o = s[load].fuse(*s[load].op.axis)
        cfg.define_knob(f"vectorize_{i}", [0, 2, 4])
        if cfg[f"vectorize_{i}"].val:
          fused_o, fused_i = s[load].split(fused, factor=cfg[f"vectorize_{i}"].val)
          s[load].vectorize(fused_i)
        fused_o, fused_i = s[load].split(fused_o, factor=cfg["tile_x"].size[2] * cfg["tile_y"].size[2])
        s[load].bind(fused_i, te.thread_axis("threadIdx.x"))

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 64, 128, 512])
    cfg.define_knob("unroll_explicit", [False, True])
    kernel_scope = rco
    s[OL].pragma(kernel_scope, 'auto_unroll_max_step', cfg['auto_unroll_max_step'].val)
    s[OL].pragma(kernel_scope, 'unroll_explicit', cfg['unroll_explicit'].val)

