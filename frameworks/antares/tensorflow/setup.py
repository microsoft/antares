#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
import shutil, os, sys

if '1.15' not in tf.version.VERSION:
  raise Exception("Current Antares plugin is for Tensorflow 1.15.x only.")

dist_path = tf.sysconfig.get_include() + '/../contrib/antares'
root_path = os.path.dirname(sys.argv[0])

if not root_path:
  root_path = '.'

try:
  os.mkdir(dist_path)
except FileExistsError:
  pass

shutil.copyfile(root_path + '/__init__.py', dist_path + '/__init__.py')
shutil.copyfile(root_path + '/main_ops.cc.in', dist_path + '/main_ops.cc.in')
shutil.copyfile(root_path + '/communicate_ops.cc', dist_path + '/communicate_ops.cc')

print("Finish Installation.")
