#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

from _common import *

x = create_variable([1024], dtype=tf.int32)

compare_ops(
  tf.one_hot(x, depth=128),
  antares.make_op('output0[N, F] = const(1.0).when([input0[N] == F], 0.0) where F in 128', [x]),
)

