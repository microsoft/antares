#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares
import os

x = tf.random.uniform([1024, 512])
y = tf.random.uniform([1024, 512])

op = antares.make_op('output0[N, M] = input0[N, M] * input1[N, M] + 1234', [x, y], server_addr=os.environ.get('ANTARES_ADDR', 'localhost:8880'))

with tf.Session() as sess:
  print(sess.run(op))
  sess.run([op] * 100)

