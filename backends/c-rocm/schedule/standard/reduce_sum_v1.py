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

    B = output
    A = B.op.input_tensors[0]
    n, k = B.op.axis[0], B.op.reduce_axis[0]

    cfg.define_split('tile_k', cfg.axis(k).length, num_outputs=2)
    ko, ki = s[B].split(k, factor=max(2, cfg['tile_k'].size[1]))
    BF = s.rfactor(B, ki)

    cfg.define_split('tile_n', cfg.axis(n).length, num_outputs=2)
    xo, xi = s[B].split(s[B].op.axis[0], factor=cfg['tile_n'].size[1])
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[B].bind(xi, te.thread_axis("threadIdx.y"))
    tx = te.thread_axis("threadIdx.x")
    s[B].bind(s[B].op.reduce_axis[0], tx)
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    s[B].set_store_predicate(tx.var.equal(0))
