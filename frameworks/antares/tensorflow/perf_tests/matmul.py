#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

from _common import *

x = create_variable([1024, 64])
y = create_variable([64, 4096])

compare_ops(
  tf.matmul(x, y),
  antares.make_op('output0[N, M] +=! input0[N, K] * input1[K, M]', [x, y]),
)

