# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tvm
import logging
import sys, time, subprocess
from tvm import autotvm
import json
import os

def schedule(attrs):
    cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]

    C = output
    A = C.op.input_tensors[0]
    n, c, h, w = C.op.axis
    p = s[C].fuse(h, w)

    cfg.define_split('tile_c', cfg.axis(c).length, num_outputs=3)
    cfg.define_split('tile_p', cfg.axis(h).length * cfg.axis(w).length, num_outputs=3)

    co, ci = s[C].split(c, factor=cfg['tile_c'].size[2])
    po, pi = s[C].split(p, factor=cfg['tile_p'].size[2])

    co, ct = s[C].split(co, factor=cfg['tile_c'].size[1])
    po, pt = s[C].split(po, factor=cfg['tile_p'].size[1])

    s[C].bind(co, tvm.thread_axis('blockIdx.x'))
    s[C].bind(po, tvm.thread_axis('blockIdx.y'))
    s[C].bind(n, tvm.thread_axis('blockIdx.z'))

    s[C].bind(ct, tvm.thread_axis('threadIdx.x'))
    s[C].bind(pt, tvm.thread_axis('threadIdx.y'))

    s[C].bind(ci, tvm.thread_axis('vthread'))
    s[C].bind(pi, tvm.thread_axis('vthread'))

    cfg.define_knob('order_case', [0, 1])
    order_list = [ci, pi] if cfg['order_case'].val == 1 else [pi, ci]
    s[C].reorder(*order_list)

