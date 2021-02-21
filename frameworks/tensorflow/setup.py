#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
import shutil, os, sys

if tf.version.VERSION < '2.4' and not tf.version.VERSION.startswith('1.15.'):
  raise Exception("Current Antares plugin is for Tensorflow >= 1.5.x / 2.4.x only.")

dist_path = tf.sysconfig.get_include() + '/../contrib/antares'
root_path = os.path.dirname(sys.argv[0])

if not root_path:
  root_path = '.'

try:
  shutil.rmtree(dist_path)
except FileNotFoundError:
  pass

os.makedirs(dist_path)

shutil.copyfile(root_path + '/__init__.py', dist_path + '/__init__.py')
shutil.copyfile(root_path + '/main_ops.cc.in', dist_path + '/main_ops.cc.in')
shutil.copyfile(root_path + '/communicate_ops.cc', dist_path + '/communicate_ops.cc')

print("Finish Installation.")
