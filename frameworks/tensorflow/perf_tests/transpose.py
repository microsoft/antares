#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

from _common import *

x = create_variable([64, 224, 224, 3], dtype=tf.float32)

compare_ops(
  tf.transpose(x, [0, 3, 1, 2]),
  antares.make_op('output0[N, C, H, W] = input0[N, H, W, C]', [x]),
)

