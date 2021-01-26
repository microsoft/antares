# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
  return ['/bin/cp', kernel_src, kernel_out]

def do_native_translation(code, **kwargs):
  code = code.replace(' __global__ ', ' ').replace(' __restrict__ ', ' ')
  brac_start = code.index('(', code.index('void ')) + 1
  brac_end = code.index(') {', brac_start)
  body_end = code.index('\n}', brac_end)

  args = [x.split() for x in code[brac_start:brac_end].split(',')]
  args_str = ', '.join([f'{arg[0]} {arg[1]}' for arg in args])
  body = code[brac_end + 3:body_end].replace('\n', '\n    ').replace('(int)blockIdx.', 'blockIdx.').replace('blockIdx.', 'axis_')

  vthread_shape = []
  for i in range(3):
    idx = body.find(f'// [thread_extent] axis_{i} =')
    if idx >= 0:
      vthread_shape.append(int(body[body.index('= ', idx) + 2:body.index('\n', idx)]))

  range_str = ', '.join([str(x) for x in vthread_shape])
  index_str = 'int %s;' % ', '.join([f'axis_{i} = _index[{i}]' for i in range(len(vthread_shape))])
  extract_args = ', '.join([f'({args[i][0]})vargs[{i}]' for i in range(len(args))])

  code = f'''
#include <CL/sycl.hpp>

extern "C" void compute_kernel(sycl::queue* q, {args_str}) {{
  q->submit([&](auto &h) {{
    h.parallel_for(sycl::range({range_str}), [=](sycl::id<{len(vthread_shape)}> _index) {{
      {index_str}{body}
    }});
  }});
}}

extern "C" void compute_kernel_vargs(sycl::queue* q, void **vargs) {{
  compute_kernel(q, {extract_args});
}}
'''
  return code
