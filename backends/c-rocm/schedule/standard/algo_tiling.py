# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm import te
import os

def plan_threads(attrs, axes):
  num_step = os.getenv('STEP', '')
  num_step = int(num_step) if num_step else 0
  if not num_step:
    return [1] * len(axes), [1] * len(axes)

  num_threads, init_threads, shape = 256, [1] * len(axes), [attrs.get_extent(ax) for ax in axes]
  for th in range(2, num_threads + 1):
    while num_threads > 1:
      unchanged = True
      for i, x in enumerate(shape):
        if x % th == 0 and num_threads % th == 0:
          num_threads //= th
          shape[i] //= th
          init_threads[i] *= th
          unchanged = False
      if unchanged:
        break
  num_vthreads, init_vthreads = 256, [1] * len(axes)
  for i, x in enumerate(shape):
    if x % 2 == 0 and num_vthreads % 2 == 0:
      num_vthreads //= 2
      shape[i] //= 2
      init_vthreads[i] *= 2
  return init_threads, init_vthreads

def schedule_branch(attrs, output, prefix):
  cfg, s = attrs.auto_config, attrs.scheduler

  init_threads, init_vthreads = plan_threads(attrs, s[output].op.axis)
  input_tensors = s[output].op.input_tensors

  data_sizes, reduce_sizes = [], []
  num_elements = 1
  for i, ax in enumerate(s[output].op.axis):
    num_elements *= attrs.get_extent(ax)
    data_sizes.append(cfg.define_split(f"{prefix}D{i}", attrs.get_extent(ax), num_outputs=4, init_vals=[[-1, 1, init_threads[i], 1], [-1, init_vthreads[i], init_threads[i], 1],[-1, 1, init_threads[i], init_vthreads[i]]]))
  for i, ax in enumerate(s[output].op.reduce_axis):
    reduce_sizes.append(cfg.define_split(f"{prefix}R{i}", attrs.get_extent(ax), num_outputs=3, init_vals=[[-1, 1, 1]]))

  num_threads, num_vthreads = 1, 1
  for i in range(len(s[output].op.axis)):
    num_threads *= data_sizes[i][2]
    num_vthreads *= data_sizes[i][1] * data_sizes[i][3]

  assert num_vthreads <= 512, "Unrecommended large vthread counts: %d" % num_vthreads
  # assert num_threads >= min(num_elements, 64), "Unrecommended small thread counts: %d" % num_threads
  assert num_threads <= attrs.device_props.max_threads_per_block, "Invalid schedule plans: num_threads(%d) > %d" % (num_threads, attrs.device_props.max_threads_per_block)

  reduce_at = cfg.define_knob(f"{prefix}RA", [x for x in range(len(s[output].op.reduce_axis))], init_vals=[0])

  output, output_local = s.cache_local(output)
  output_local_rv_o_o, output_local_rv_o_i, output_local_rv_i = cfg.apply_split(s, output_local, output_local.op.reduce_axis[reduce_at], reduce_sizes[reduce_at])

  local_slices = [list(cfg.apply_split(s, output_local, output_local.op.axis[i], [-1, 1] + data_sizes[i][1:])) for i in range(len(output_local.op.axis))]
  zero, first, second, third, fourth = [x[0] for x in local_slices], [x[1] for x in local_slices], [x[2] for x in local_slices], [x[3] for x in local_slices], [x[4] for x in local_slices]
  s[output_local].reorder(*(zero + first + second + [output_local_rv_o_o, output_local_rv_o_i] + third + [output_local_rv_i] + fourth))

  data_slices = [list(cfg.apply_split(s, output, output.op.axis[i], data_sizes[i])) for i in range(len(output.op.axis))]

  first, second, third, fourth = [x[0] for x in data_slices], [x[1] for x in data_slices], [x[2] for x in data_slices], [x[3] for x in data_slices]

  s[output].reorder(*(first + second + third + fourth))
  s[output_local].compute_at(s[output], third[-1])

  s[output].bind(s[output].fuse(*first), te.thread_axis("blockIdx.x"))
  s[output].bind(s[output].fuse(*second), te.thread_axis("vthread"))
  s[output].bind(s[output].fuse(*third), te.thread_axis("threadIdx.x"))

  load_stage = []
  for load in input_tensors:
    load_stage.append(s.cache_read(load, 'shared', [output_local]))
    s[load_stage[-1]].compute_at(s[output_local], output_local_rv_o_o)

  for i, load in enumerate(load_stage):
    fused_o = s[load].fuse(*s[load].op.axis)
    val = 1 ## cfg.define_knob(f"{prefix}V{i}", [1, 2, 4] if not attrs.backend.startswith('c-hlsl_') else [1])
    fused_o, fused_i = s[load].split(fused_o, factor=val)
    s[load].vectorize(fused_i)
    fused_o, fused_i = s[load].split(fused_o, factor=num_threads)
    s[load].bind(fused_i, te.thread_axis("threadIdx.x"))

  # unroll
  unroll_step = cfg.define_knob(f"{prefix}S", [1, 4, 32, 512])
  unroll_explicit = cfg.define_knob(f"{prefix}U", [False, True])
  kernel_scope = zero[0]
  s[output_local].pragma(kernel_scope, 'auto_unroll_max_step', unroll_step)
  s[output_local].pragma(kernel_scope, 'unroll_explicit', unroll_explicit)
