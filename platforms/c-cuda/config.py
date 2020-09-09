# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess, os

def get_execution_parallism():
    return len(subprocess.getoutput('ls /dev/nvidia[0-9]* 2>/dev/null').split())

def get_compile_kernel_args(kernel_src, kernel_out, device_props):
  code_arch = device_props.compute_version.replace('.', '')
  assert 0 == os.system('ln -sf %s %s.cu' % (kernel_src, kernel_src))
  assert 0 == os.system(' '.join(['/usr/local/cuda/bin/nvcc', kernel_src + '.cu', '--ptx', '-O2', '-gencode', 'arch=compute_%s,code=sm_%s' % (code_arch, code_arch), '-o', '%s.ptx' % kernel_src]))
  return ['/usr/local/cuda/bin/nvcc', kernel_src + '.ptx', '--fatbin', '-O2', '-gencode', 'arch=compute_%s,code=sm_%s' % (code_arch, code_arch), '-o', kernel_out]

def do_native_translation(code, **kwargs):
  code = '#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n\n' + kwargs['attrs'].blend + '\n' + code
  return code
