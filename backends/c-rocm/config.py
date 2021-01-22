# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return len(subprocess.getoutput('/opt/rocm/bin/rocm_agent_enumerator | grep -v gfx000').split())

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
  major, minor = device_props.compute_version.split('.')
  code_arch = str(int(major) * 100 + int(minor))
  return ['/opt/rocm/bin/hipcc', kernel_src, '--amdgpu-target=gfx' + code_arch, '--genco', '-Wno-ignored-attributes', '-O2', '-o', kernel_out]

def do_native_translation(code, **kwargs):
  def parse_launch_bounds(code):
    func_arr = code.split('extern "C" __global__ ')
    for i in range(1, len(func_arr)):
      axis_map = dict()
      lines = func_arr[i].split('\n')
      for it in lines:
        if it.startswith('  // [thread_extent] '):
          words = it.split(' ')
          nthread = int(words[-1])
          axis = words[-3]
          if axis in axis_map:
            if axis_map[axis] != nthread:
              assert(False)
          else:
            axis_map[axis] = nthread
      block_bound = axis_map.get('threadIdx.x', 1) * axis_map.get('threadIdx.y', 1) * axis_map.get('threadIdx.z', 1)
      func_arr[i] = 'extern "C" __global__ __launch_bounds__(%d) %s' % (block_bound, func_arr[i])
    code = ''.join(func_arr)
    return code

  code = parse_launch_bounds(code)
  return '#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n\n' + kwargs['attrs'].blend + '\n' + code
