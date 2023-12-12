#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

input0 = tf.get_variable('input0', [1024 * 512], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)
input1 = tf.get_variable('input1', [1024 * 512], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

op = antares.make_op(ir='output0[N] = input0[N] + input1[N]; output1[N] = input0[N].call(`exp`); output2[N] = input1[N] + output1[N];', extra_outputs=['output0', 'output1', 'output2'], feed_dict={'input0': input0, 'input1': input1}).tune(step=100, use_cache=True, timeout=600).emit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  print('The result of tensor `%s` is:\n%s' % (op, sess.run(op)))

