#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil, os, sys
from setuptools import setup

if len(sys.argv) <= 1:
  sys.argv += ['install']

root_path = os.path.dirname(sys.argv[0])

if not root_path:
  root_path = '.'

for tree in ('antares_custom_op.egg-info', 'build', 'dist'):
  try:
    shutil.rmtree(f'{root_path}/{tree}')
  except:
    pass

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, IS_HIP_EXTENSION

dist_path = os.path.join(torch.__path__[0], 'contrib/antares')

os.chdir(root_path)
root_path = '.'
sys.dont_write_bytecode = False

try:
  os.mkdir(dist_path)
except FileExistsError:
  pass

shutil.copyfile(root_path + '/custom_op.py', dist_path + '/custom_op.py')
shutil.copyfile(root_path + '/../../graph_evaluator/execute_module.hpp', dist_path + '/execute_module.hpp')

cpp_flags = [f'-I{dist_path}', '-Wno-sign-compare', '-Wno-address', '-Wno-unused-value']

try:
  if torch.cuda.is_available():
    shutil.copyfile(root_path + '/../../backends/c-rocm/include/backend.hpp', dist_path + '/backend.hpp')
    backend = 'c-cuda' if not IS_HIP_EXTENSION else 'c-rocm'
    cpp_flags += [f'-D__BACKEND__=\"{backend}\"', '-DANTARES_CUDA' if not IS_HIP_EXTENSION else '-DANTARES_ROCM']
    ext = CUDAExtension(
            'antares_custom_op',
            ['main_ops.cc'],
            libraries=['cuda'] if not IS_HIP_EXTENSION else [],
            extra_compile_args={'cxx': cpp_flags, 'nvcc': cpp_flags},
        )
  else:
    raise
except:
  shutil.copyfile(root_path + '/../../backends/c-mcpu/include/backend.hpp', dist_path + '/backend.hpp')
  backend = 'c-mcpu_avx512' if os.system("grep -r '\\bavx512' /proc/cpuinfo >/dev/null") == 0 else 'c-mcpu'
  cpp_flags += [f'-D__BACKEND__=\"{backend}\"', '-DANTARES_MCPU']
  ext = CppExtension('antares_custom_op', ['main_ops.cc'], extra_compile_args={'cxx': cpp_flags}, language='c++')

setup(
    name='antares_custom_op',
    ext_modules=[ext,],
    cmdclass={
        'build_ext': BuildExtension
    })

print("Finish Installation.")
