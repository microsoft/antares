# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import tvm
import logging
import sys, time, subprocess
from tvm import autotvm
# import topi
import json
# from topi.util import get_const_tuple
import os


def schedule(attrs):
    cfg, s, output = attrs.auto_config, attrs.scheduler, attrs.outputs[0]
    th_vals, rd_vals = [attrs.get_extent(x) for x in output.op.axis], [attrs.get_extent(x) for x in output.op.reduce_axis]

    C = output
    A, B = C.op.input_tensors

    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # storage_align params
    factor = 16
    offset = 8

    layout = 'NN'
    for opt in attrs.options:
      if opt.startswith('layout/'):
        layout = opt[len('layout/'):]
        break

    '''
    if dtype == 'int8':
      factor = 32
      offset = 16
    '''
    # create cache stages
    AA = s.cache_read(A, "shared", [C])
    if (layout == "NN" or layout == "TN"):
      s[AA].storage_align(AA.op.axis[0], factor, offset)
    AL = s.cache_read(AA, "local", [C])
    BB = s.cache_read(B, "shared", [C])
    if (layout == "TT" or layout == "NT"):
      s[BB].storage_align(BB.op.axis[0], factor, offset)
    BL = s.cache_read(BB, "local", [C])
    CL = s.cache_write(C, "local")

    #autotvm search space definition
    cfg.define_knob("bx", [2, 4, 8])
    cfg.define_knob("by", [16, 32, 64])
    cfg.define_knob("step_k", [8, 16, 32])
    cfg.define_knob("v", [4, 8])
    by = cfg['by'].val
    bx = cfg['bx'].val
    step_k = cfg['step_k'].val
    v = cfg['v'].val

    # thread tile
    TX, TY = 8, 1

    # warp tile
    cfg.define_knob("warp_m", [16, 8, 32])
    warp_tile_m = cfg['warp_m'].val # it could be 8, 16, 32 on CUDA version >= 10.0
    warp_tile_k = 16 # it must be 16
    # block tile
    tile_x = bx * TX
    tile_y = by * TY

    yo, ty = s[C].split(y, tile_y)
    ty, yi = s[C].split(ty, TY)

    # schedule for C stage
    xo, xi = s[C].split(x, tile_x)
    WX = min(warp_tile_m, tile_x)
    tz, xi = s[C].split(xi, WX)
    tx, xi = s[C].split(xi, TX)
    s[C].reorder(yo, xo, tz, ty, tx, yi, xi)
    s[C].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[C].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[C].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[C].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

    # schedule for CL stage
    ko, ki = s[CL].split(k, step_k * warp_tile_k)
    kl, ki = s[CL].split(ki, warp_tile_k)
    s[CL].compute_at(s[C], tx)
    yo, xo = CL.op.axis
    s[CL].reorder(ko, kl, ki, yo, xo)

    # schedule for AA stage
    s[AA].compute_at(s[CL], ko)
    xo, xi = s[AA].split(s[AA].op.axis[1], factor=bx*v)
    tz, tx = s[AA].split(xi, factor=(WX//TX)*v)
    tx, vec = s[AA].split(tx, factor=v)
    fused = s[AA].fuse(s[AA].op.axis[0], xo)
    _, ty = s[AA].split(fused, factor=by)
    s[AA].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[AA].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[AA].bind(tx, tvm.thread_axis("threadIdx.x"))
    # vectorization is very important for float16/int8 inputs
    s[AA].vectorize(vec)

    # schedule for BB stage
    s[BB].compute_at(s[CL], ko)
    xo, xi = s[BB].split(s[BB].op.axis[1], factor=bx*v)
    tz, tx = s[BB].split(xi, factor=(WX//TX)*v)
    tx, vec = s[BB].split(tx, factor=v)
    fused = s[BB].fuse(s[BB].op.axis[0], xo)
    _, ty = s[BB].split(fused, factor=by)
    s[BB].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[BB].bind(tz, tvm.thread_axis("threadIdx.z"))
    s[BB].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[BB].vectorize(vec)

    s[AL].compute_at(s[CL], kl)
    s[BL].compute_at(s[CL], kl)

    s[CL].pragma(ko, 'tensor_core')

