# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from tvm import te
import logging
import sys, time, subprocess


import json
import os


def schedule(attrs):
    cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
    th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

    C = output
    A, B = C.op.input_tensors

    AA = s.cache_read(A, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    BL = s.cache_read(BB, "local", [C])
    if C.op in s.outputs:
      CC = s.cache_write(C, "local")
    else:
      s[C].set_scope('local')
      CC, C = C, s.outputs[0].output(0)

    axes = C.op.axis
    y, x = axes[-2], axes[-1]
    b = s[C].fuse(*axes[:-2])
    k = CC.op.reduce_axis[0]

    cfg.flop = float(np.product(th_vals) * rd_vals[0] * 2.0)

    cfg.define_split('tile_k', cfg.axis(k), num_outputs=3)
    ko, kt, ki = cfg['tile_k'].apply(s, CC, k)

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    s[C].bind(b, te.thread_axis('blockIdx.z'))

    cfg.define_split('tile_y', cfg.axis(y), num_outputs=4)
    cfg.define_split('tile_x', cfg.axis(x), num_outputs=4)

    by, tyz, ty, yi = cfg['tile_y'].apply(s, C, y)
    bx, txz, tx, xi = cfg['tile_x'].apply(s, C, x)

    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].bind(tyz, te.thread_axis('vthread'))
    s[C].bind(txz, te.thread_axis('vthread'))
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)

    s[CC].compute_at(s[C], tx)

    # b, yo, xo = CC.op.axis
    s[CC].reorder(ko, kt, *CC.op.axis, ki)
    s[CC].unroll(kt)

    for stage in [AL, BL]:
        s[stage].compute_at(s[CC], kt)
        # _, xi = s[stage].split(stage.op.axis[1], factor=4)
        # s[stage].vectorize(xi)
        s[stage].double_buffer()

    cfg.define_knob('vectorize', [False, True] if attrs.backend != 'c-hlsl_win64' else [False])
    # cfg.define_knob('storage_align', [16, 48])
    for stage in [AA, BB]:
        # s[stage].storage_align(s[stage].op.axis[0],
        #                        cfg['storage_align'].val, 0)
        s[stage].compute_at(s[CC], ko)

        fused = s[stage].fuse(*s[stage].op.axis)
        ty, tx = s[stage].split(fused, nparts=cfg['tile_y'].size[2])
        tx, xi = s[stage].split(tx, nparts=cfg['tile_x'].size[2])
        _, xi = s[stage].split(xi, factor=4)

        s[stage].bind(ty, thread_y)
        s[stage].bind(tx, thread_x)
        if cfg['vectorize'].val:
            s[stage].vectorize(xi)
        s[stage].double_buffer()

    s[C].pragma(by, 'auto_unroll_max_step', 125)
    s[C].pragma(by, 'unroll_explicit', False)

