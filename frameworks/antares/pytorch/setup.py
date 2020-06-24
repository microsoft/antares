#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import shutil, os, sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist_path = os.path.join(torch.__path__[0], 'contrib/antares')
root_path = os.path.dirname(sys.argv[0])

if not root_path:
  root_path = '.'

os.chdir(root_path)
root_path = '.'
sys.dont_write_bytecode = False

try:
  os.mkdir(dist_path)
except FileExistsError:
  pass

shutil.copyfile(root_path + '/custom_op.py', dist_path + '/custom_op.py')

is_cuda = (os.system('ldd %s/lib/libtorch.so 2>/dev/null | grep -e libcudart >/dev/null' % torch.__path__[0]) == 0)

setup(
    name='antares_custom_op',
    ext_modules=[
        CUDAExtension(
            'antares_custom_op',
            ['main_ops.cc.cu'],
            libraries=['cuda'] if is_cuda else []
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

print("Finish Installation.")
