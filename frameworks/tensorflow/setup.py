#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
import shutil, os, sys

if tf.version.VERSION < '2.3' and not tf.version.VERSION.startswith('1.15.'):
  raise Exception("Current Antares plugin is for Tensorflow >= 1.5.x / 2.3.x only.")

dist_path = tf.sysconfig.get_include() + '/../contrib/antares'
root_path = os.path.dirname(__file__)

if not root_path:
  root_path = '.'

try:
  shutil.rmtree(dist_path)
except FileNotFoundError:
  pass

os.makedirs(dist_path)

shutil.copyfile(root_path + '/__init__.py', dist_path + '/__init__.py')
shutil.copyfile(root_path + '/communicate_ops.cc', dist_path + '/communicate_ops.cc')

shutil.copyfile(root_path + '/main_ops.cc.in', dist_path + '/main_ops.cc.in')
shutil.copyfile(root_path + '/../../graph_evaluator/execute_module.hpp', dist_path + '/execute_module.hpp')

if tf.test.is_built_with_gpu_support():
  shutil.copyfile(root_path + '/../../backends/c-rocm/include/backend.hpp', dist_path + '/backend.hpp')
elif os.system('which dpcpp >/dev/null') == 0:
  shutil.copyfile(root_path + '/../../backends/c-sycl_intel/include/backend.hpp', dist_path + '/backend.hpp')
else:
  shutil.copyfile(root_path + '/../../backends/c-mcpu/include/backend.hpp', dist_path + '/backend.hpp')

print("Finish Installation libraries to: %s" % os.path.realpath(dist_path))
