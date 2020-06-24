#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

from _common import *

x = create_variable([1024, 3072], dtype=tf.float32)

compare_ops(
  tf.reduce_sum(x, axis=1),
  antares.make_op('output0[N] +=! input0[N, M]', [x]),
)

