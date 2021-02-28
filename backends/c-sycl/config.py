# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
import numpy as np

def get_execution_parallism():
  return 1

using_buffer_accessor = False  # Intel CPU device is slow when using_buffer_accessor

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset

  if using_buffer_accessor:
    expand_args = '\n  '.join([f'auto __args_{i} = cl::sycl::buffer<{x[0]}>(({x[0]}*)__args[{i}], cl::sycl::range<1>({int(np.product(x[2]["shape"]))}));' for i, x in enumerate(in_args + out_args)])
    expand_accs = '\n    '.join([f'auto __accs_{i} = __args_{i}.get_access<cl::sycl::access::mode::{"read" if i < len(in_args) else "write"}>(cgh);' for i, x in enumerate(in_args + out_args)]) + '\n'
    expand_ptrs = '\n      '.join([f'{x[0]}* {x[1]} = __accs_{i}.get_pointer();' for i, x in enumerate(in_args + out_args)]) + '\n'
  else:
    expand_args = ' '.join([f'{x[0]}* {x[1]} = ({x[0]}*)__args[{i}];' for i, x in enumerate(in_args + out_args)])
    expand_accs = expand_ptrs = ''

  def get_extent(key, defval=1):
    str_pat = f'// [thread_extent] {key} = '
    idx = body.find(str_pat)
    if idx >= 0:
      return int(body[idx+len(str_pat):body.index('\n', idx)])
    return defval

  group_shared = []
  parsed_lines, body = [], body.split('\n')
  for line in body:
    simple_line = line.strip()
    if not simple_line.startswith('__shared__ '):
      parsed_lines.append(line)
      continue
    _, type, data = simple_line.split()
    name, size_str = data[:-2].split('[')
    parsed_lines.append(f'{line[0:len(line)-len(simple_line)]}{type}* {name} = __accessor_{name}.get_pointer();');
    group_shared.append(f'sycl::accessor<{type}, 1, sycl::access::mode::read_write, sycl::access::target::local> __accessor_{name}(sycl::range<1>({size_str}), cgh);');
  body = '\n'.join(parsed_lines)
  group_shared = '    \n'.join(group_shared)
  del parsed_lines

  body = body.replace('Idx.', 'Idx_').replace('__syncthreads()', '_item.barrier(cl::sycl::access::fence_space::global_and_local);').replace('\n', '\n    ')
  body = body.replace('(make_int4)', 'make_int4').replace('(make_int2)', 'make_int2')
  index_str = 'const int blockIdx_x = _item.get_group(0), blockIdx_y = _item.get_group(1), blockIdx_z = _item.get_group(2), threadIdx_x = _item.get_local_id(0), threadIdx_y = _item.get_local_id(1), threadIdx_z = _item.get_local_id(2);'

  lds = [get_extent('threadIdx_x'), get_extent('threadIdx_y'), get_extent('threadIdx_z')]
  gds = [get_extent('blockIdx_x') * lds[0], get_extent('blockIdx_y') * lds[1], get_extent('blockIdx_z') * lds[2]]

  full_body = f'''#include <math.h>
#include <algorithm>
#include <CL/sycl.hpp>
{kwargs['attrs'].blend}

#ifndef __MAKE_DATA_ARRAY__
#define __MAKE_DATA_ARRAY__

struct int2 {{ int x, y; }};
struct int4 {{ int x, y, z, w; }};
struct float2 {{ float x, y; }};
struct float4 {{ float x, y, z, w; }};

#define make_int4(x, y)  (int2{{x, y}})
#define make_float4(x, y)  (float4{{x, y}})
#define make_int2(x, y)  (int2{{x, y}})
#define make_float2(x, y)  (float2{{x, y}})

#endif

extern "C" void {kernel_name}(sycl::queue* q, void **__args) {{
  {expand_args}

  using namespace std;

  q->submit([&](auto &cgh) {{
    {expand_accs}

    {group_shared}

    cgh.parallel_for(cl::sycl::nd_range<3>(cl::sycl::range<3>({str(gds)[1:-1]}), cl::sycl::range<3>({str(lds)[1:-1]})), [=](cl::sycl::nd_item<3> _item) {{
      {expand_ptrs}
      {index_str}

      {body}
    }});
  }});
}}
'''
  return full_body
