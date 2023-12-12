#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

from _common import *

x = create_variable([1024, 64], dtype=tf.float32)

compare_ops(
  tf.broadcast_to(tf.reshape(x, [1024, 64, 1]), shape=[1024, 64, 16]),
  antares.make_op('output0[N, M, K] = input0[N, M] where K in 16', [x]),
)

