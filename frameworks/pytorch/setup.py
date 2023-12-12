#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil, os, sys
from setuptools import setup

if len(sys.argv) <= 1:
  sys.argv += ['install', '--user']

root_path = os.path.dirname(__file__)
root_path = root_path or '.'

os.chdir(root_path)
root_path = os.getcwd()
sys.dont_write_bytecode = False

assert 'BACKEND' in os.environ, "Please set BACKEND type in environment variables."
backend = os.environ['BACKEND']

package_name = 'antares_custom_torch_v2_' + backend.replace('-', '_')

for tree in (f'{package_name}.egg-info', 'build', 'dist'):
  try:
    shutil.rmtree(f'./{tree}')
  except:
    pass

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, IS_HIP_EXTENSION

cpp_flags = ['-I' + os.path.join(torch.__path__[0]), '-Wno-sign-compare', '-Wno-address', '-Wno-unused-value', '-Wno-strict-aliasing']

if 'cuda' in backend or 'rocm' in backend:
  cpp_flags += [f'-D__BACKEND__=\"{backend}\"']
  cpp_flags += [f'-I{root_path}/../../backends/{backend}/include', f'-I{root_path}/../../graph_evaluator']
  ext = CUDAExtension(
            package_name,
            ['main_ops.cc'],
            libraries=[':libcuda.so.1'] if not IS_HIP_EXTENSION else [],
            extra_compile_args={'cxx': cpp_flags, 'nvcc': cpp_flags},
        )
else:
  cpp_flags += [f'-D__BACKEND__=\"{backend}\"']
  cpp_flags += [f'-I{root_path}/../../backends/{backend}/include', f'-I{root_path}/../../graph_evaluator']
  ext = CppExtension(package_name, ['main_ops.cc'], extra_compile_args={'cxx': cpp_flags}, language='c++')

setup(
    name=package_name,
    ext_modules=[ext,],
    cmdclass={
        'build_ext': BuildExtension
    })

print("Finish Installation.")
