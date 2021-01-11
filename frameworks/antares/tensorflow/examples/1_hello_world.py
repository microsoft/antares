#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

x = tf.get_variable('x', [128, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)
y = tf.get_variable('y', [1024, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

op = antares.make_op(ir='dot_0[N, M] +=! data[N, K] * weight[K, M]', feed_dict={'data': x, 'weight': y}).tune(step=100, use_cache=True, timeout=600).emit()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print('The result of tensor `%s` is:\n%s' % (op, sess.run(op)))

