#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares
import os

x = tf.random.uniform([1024, 512])

op = antares.make_op('reduce_sum_0[N] +=! data[N, M]', {'data': x}, server_addr=os.environ.get('ANTARES_ADDR', 'localhost:8880'))

with tf.Session() as sess:
  print('The result of tensor `%s` is:\n%s' % (op._output_names[0], sess.run(op)))
  sess.run([op] * 100)

