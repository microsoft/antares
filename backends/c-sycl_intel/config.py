# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os
from common import backend

def to_search_space(ast_seq, input_dict, output_dict):
  from antares.default_codegen import codegen
  from antares.common import AntaresGlobal
  codegen(ast_seq, input_dict, output_dict, {}, space_only=True)
  space = AntaresGlobal.auto_config.get_config_space()
  return space

def to_kernel_slices(compute_graph, best_config):
  from antares.default_codegen import codegen
  return codegen(*compute_graph, best_config)

def get_execution_parallism():
  if backend == 'c-sycl_intel':
    return 1
  num_gpus = len(subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split())
  num_gpus = num_gpus if num_gpus > 0 else 1
  return num_gpus

def do_native_translation_v2(codeset, **kwargs):
  kernel_name, in_args, out_args, body = codeset

  expand_ins = [f'const auto* {x[1]} = ({x[0]}* __restrict)__args[{i}];' for i, x in enumerate(in_args)]
  expand_outs = [f'auto* {x[1]} = ({x[0]}* __restrict)__args[{i + len(in_args)}];' for i, x in enumerate(out_args)]
  expand_args = ' '.join(expand_ins + expand_outs)
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
  group_shared = '\n    '.join(group_shared)
  del parsed_lines

  body = body.replace('Idx.', 'Idx_')
  body = body.replace('__syncthreads()', '_item.barrier(cl::sycl::access::fence_space::local_space);').replace('\n', '\n    ')

  # Reversed order in dim configs
  index_str = 'const int blockIdx_x = _item.get_group(2), blockIdx_y = _item.get_group(1), blockIdx_z = _item.get_group(0), threadIdx_x = _item.get_local_id(2), threadIdx_y = _item.get_local_id(1), threadIdx_z = _item.get_local_id(0);'
  lds = [get_extent('threadIdx_z'), get_extent('threadIdx_y'), get_extent('threadIdx_x')]
  gds = [get_extent('blockIdx_z') * lds[0], get_extent('blockIdx_y') * lds[1], get_extent('blockIdx_x') * lds[2]]

  full_body = f'''#include <math.h>
#include <algorithm>
#include <CL/sycl.hpp>
{kwargs['attrs'].blend}

#ifndef __SYCL_COMMON_MACRO__
#define __SYCL_COMMON_MACRO__

#define make_int4(x, y, z, w)  (int4{{x, y, z, w}})
#define make_int2(x, y)  (int2{{x, y}})

#define __STORE_ITEM_0__(t, out, ido, in, idi) *(t*)(out + ido) = *(t*)(in + idi)
#define __STORE_ITEM_1__(t, out, ido, in, idi)
#define __STORE_ITEM_2__(t, out, ido, in, idi)
#define __STORE_ITEM_3__(t, out, ido, in, idi)

#define __ITEM_0_OF__(v) (v).x()
#define __ITEM_1_OF__(v) (v).y()
#define __ITEM_2_OF__(v) (v).z()
#define __ITEM_3_OF__(v) (v).w()

#endif

extern "C" void {kernel_name}(sycl::queue* q, void **__args) {{
  {expand_args}

  using namespace cl::sycl;

  q->submit([&](auto &cgh) {{
    {group_shared}
    {expand_accs}
    cgh.parallel_for(cl::sycl::nd_range<3>(cl::sycl::range<3>({str(gds)[1:-1]}), cl::sycl::range<3>({str(lds)[1:-1]})), [=](cl::sycl::nd_item<3> _item) {{
      {expand_ptrs}
      {index_str}

      {body}
    }});
  }});
}}
'''
  return full_body
