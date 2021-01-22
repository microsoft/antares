# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
  return len(subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split())

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
  code_arch = device_props.compute_version.replace('.', '')
  assert 0 == os.system('ln -sf %s %s.cu' % (kernel_src, kernel_src))
  # os.system(' '.join(['/usr/local/cuda/bin/nvcc', kernel_src + '.cu', '--ptx', '-O2', '-gencode', 'arch=compute_%s,code=sm_%s' % (code_arch, code_arch), '-o', '%s.ptx' % kernel_src]))
  return ['/usr/local/cuda/bin/nvcc', kernel_src + '.cu', '--fatbin', '-O2', '-gencode', 'arch=compute_%s,code=sm_%s' % (code_arch, code_arch), '-o', kernel_out]

def do_native_translation(code, **kwargs):
  headers = ['#include <cuda_runtime.h>', '#include <cuda_fp16.h>', '#include <mma.h>']
  code = '''
#ifndef __HALF_COMPARE_EX__
#define __HALF_COMPARE_EX__
inline __device__ half max(half x, half y) { return x > y ? x : y; }
inline __device__ half min(half x, half y) { return x < y ? x : y; }
#endif

''' + code
  code = '\n'.join(headers) + '\n' + kwargs['attrs'].blend + code
  return code
