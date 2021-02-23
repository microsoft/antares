# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return 1

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset
  expand_args = ' '.join([f'{x[0]}* {x[1]} = ({x[0]}*)__args[{i}];' for i, x in enumerate(in_args + out_args)])

  body = body.replace('blockIdx.', 'axis_').replace('\n', '\n    ')
  thread_extents = []
  for i in range(3):
    kword = f'// [thread_extent] axis_{i} = '
    idx = body.find(kword)
    if idx >= 0:
      thread_extents.append(int(body[idx+len(kword):body.index('\n', idx+len(kword))]))
  range_str = ', '.join([str(x) for x in thread_extents])
  index_str = 'int %s;' % ', '.join([f'axis_{i} = _index[{i}]' for i in range(len(thread_extents))])

  full_body = f'''#include <math.h>
#include <algorithm>
#include <CL/sycl.hpp>
{kwargs['attrs'].blend}

extern "C" void {kernel_name}(sycl::queue* q, void **__args) {{
  {expand_args}
  using namespace std;

  q->submit([&](auto &h) {{
    h.parallel_for(sycl::range({range_str}), [=](sycl::id<{len(thread_extents)}> _index) {{
      {index_str}
      {body}
    }});
  }});
}}
'''
  return full_body
