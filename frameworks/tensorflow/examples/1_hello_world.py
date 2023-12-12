#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

x = tf.get_variable('x', [128, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)
y = tf.get_variable('y', [1024, 1024], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

op = antares.make_op(ir='dot_0[N, M] +=! data[N, K] * weight[K, M]', feed_dict={'data': x, 'weight': y}).tune(step=100, use_cache=True, timeout=600).emit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  print('The result of tensor `%s` is:\n%s' % (op, sess.run(op)))

