#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares
import time

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

x = tf.get_variable('x', [64, 28 * 28], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

def create_param(name, shape):
  return tf.get_variable(name, shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001), trainable=True)

w0 = create_param('dense_w0', [28 * 28, 512])
b0 = create_param('dense_b0', [512])
w1 = create_param('dense_w1', [512, 512])
b1 = create_param('dense_b1', [512])
w2 = create_param('dense_w2', [512, 10])
b2 = create_param('dense_b2', [10])

tf_out = x
tf_out = tf.add(tf.matmul(tf_out, w0), b0)
tf_out = tf.nn.relu(tf_out)
tf_out = tf.add(tf.matmul(tf_out, w1), b1)
tf_out = tf.nn.relu(tf_out)
tf_out = tf.add(tf.matmul(tf_out, w2), b2)

out = x
out = antares.make_op(ir='''
  data_0[N, M] +=!  data[N, K] * weight_0[K, M];
  data_0_bias[N, K] = data_0[N, K] + bias_0[K];
  data_1[N, K] =   data_0_bias[N, K].call(`max`, [0.0]);
  data_2[N, M] +=!  data_1[N, K] * weight_1[K, M];
  data_2_bias[N, K] = data_2[N, K] + bias_1[K];
  data_3[N, K] =   data_2_bias[N, K].call(`max`, [0.0]);
  data_4[N, M] +=!  data_3[N, K] * weight_2[K, M];
  data_5[N, K] =   (data_4[N, K] + bias_2[K]);
''', feed_dict={'data': x, 'weight_0': w0, 'weight_1': w1, 'weight_2': w2, 'bias_0': b0, 'bias_1': b1, 'bias_2': b2}).tune(step=200, use_cache=True, timeout=600).emit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  print('[Tensorflow Result]')
  print(sess.run([tf_out]))
  print('[Antares Result]')
  print(sess.run([out]))

###### Result on AMD Radeon-7:
#   TF-native Average Time per Run:                    0.000643362 (sec)
#   TVM-Ansor (step=200) Average Time per Run:         0.000459301 (sec)
#   Antares-OpEvo (step=200) Average Time per Run:     0.000218356 (sec)

###### Result on NVIDIA Volta-100:
#   TF-native Average Time per Run:                    0.000414934 (sec)
#   TVM-Ansor (step=200) Average Time per Run:         0.000128158 (sec)
#   Antares-OpEvo (step=200) Average Time per Run:     0.000080845 (sec)
