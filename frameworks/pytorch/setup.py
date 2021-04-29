#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil, os, sys
from setuptools import setup

if len(sys.argv) <= 1:
  sys.argv += ['install']

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, IS_HIP_EXTENSION

dist_path = os.path.join(torch.__path__[0], 'contrib/antares')
root_path = os.path.dirname(sys.argv[0])

if not root_path:
  root_path = '.'

for tree in ('antares_custom_op.egg-info', 'build', 'dist'):
  try:
    shutil.rmtree(f'{root_path}/{tree}')
  except:
    pass

os.chdir(root_path)
root_path = '.'
sys.dont_write_bytecode = False

try:
  os.mkdir(dist_path)
except FileExistsError:
  pass

shutil.copyfile(root_path + '/custom_op.py', dist_path + '/custom_op.py')
shutil.copyfile(root_path + '/../../graph_evaluator/execute_module.hpp', dist_path + '/execute_module.hpp')

if torch.cuda.is_available():
  shutil.copyfile(root_path + '/../../backends/c-rocm/include/backend.hpp', dist_path + '/backend.hpp')
  is_cuda = not IS_HIP_EXTENSION
else:
  shutil.copyfile(root_path + '/../../backends/c-mcpu/include/backend.hpp', dist_path + '/backend.hpp')
  is_cuda = False

setup(
    name='antares_custom_op',
    ext_modules=[
        CUDAExtension(
            'antares_custom_op',
            ['main_ops.cc.cu'],
            libraries=['cuda'] if is_cuda else [],
            extra_compile_args={'cxx': [f'-I{dist_path}'], 'nvcc': [f'-I{dist_path}']},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

print("Finish Installation.")
