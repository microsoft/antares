#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()


def create_param(name, shape):
  return tf.get_variable(name, shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.001), trainable=True)

input_tensor = tf.get_variable('input_tensor', [64, 3, 227, 227], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

feed_dict={
  'input_tensor': input_tensor,
  'const_0_': create_param('const_0_', [11, 11, 3, 64]),
  'const_1_': create_param('const_1_', [5, 5, 64, 192]),
  'const_2_': create_param('const_2_', [3, 3, 192, 384]),
  'const_3_': create_param('const_3_', [3, 3, 384, 256]),
  'const_4_': create_param('const_4_', [3, 3, 256, 256]),
  'const_5_': create_param('const_5_', [9216, 4096]),
  'const_6_': create_param('const_6_', [4096, 4096]),
  'const_7_': create_param('const_7_', [4096, 1000]),
}

conv_0 = tf.nn.conv2d(input_tensor, filters=feed_dict['const_0_'], strides=[1, 1, 4, 4], padding='VALID', data_format='NCHW')
mpool_0 = tf.nn.max_pool(tf.nn.relu(conv_0), ksize=3, strides=2, padding='VALID', data_format='NCHW')
conv_1 = tf.nn.conv2d(mpool_0, filters=feed_dict['const_1_'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
mpool_1 = tf.nn.max_pool(tf.nn.relu(conv_1), ksize=3, strides=2, padding='VALID', data_format='NCHW')
conv_2 = tf.nn.conv2d(mpool_1, filters=feed_dict['const_2_'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
conv_3 = tf.nn.conv2d(tf.nn.relu(conv_2), filters=feed_dict['const_3_'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
conv_4 = tf.nn.conv2d(tf.nn.relu(conv_3), filters=feed_dict['const_4_'], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
mpool_2 = tf.nn.max_pool(tf.nn.relu(conv_4), ksize=3, strides=2, padding='VALID', data_format='NCHW')
reshape_0 = tf.reshape(mpool_2, [input_tensor.shape[0], -1])
dense_0 = tf.matmul(reshape_0, feed_dict['const_5_'])
dense_1 = tf.matmul(tf.nn.relu(dense_0), feed_dict['const_6_'])
dense_2 = tf.matmul(tf.nn.relu(dense_1), feed_dict['const_7_'])
output_logits_tf = dense_2

output_logits = antares.make_op(ir=f'''
  conv_0[N, F, HO, WO] +=! input_tensor[N, C, HO * 4 + KH, WO * 4 + KW] * const_0_[KH, KW, C, F] where HO in 55, WO in 55;
  mpool_0[N, C, HO, WO] >=! conv_0[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 27, WO in 27, KH in 3, KW in 3;
  conv_1[N, F, HO, WO] +=! mpool_0[N, C, -2 + HO + KH, -2 + WO + KW].when([-2 + HO + KH >= 0, -2 + HO + KH < 27, -2 + WO + KW >= 0, -2 + WO + KW < 27], 0.0) * const_1_[KH, KW, C, F] where HO in 27, WO in 27;
  mpool_1[N, C, HO, WO] >=! conv_1[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 13, WO in 13, KH in 3, KW in 3;
  conv_2[N, F, HO, WO] +=! mpool_1[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_2_[KH, KW, C, F] where HO in 13, WO in 13;
  conv_2_relu[N, F, HO, WO] = conv_2[N, F, HO, WO].call(`max`, [0.0]);
  conv_3[N, F, HO, WO] +=! conv_2_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_3_[KH, KW, C, F] where HO in 13, WO in 13;
  conv_3_relu[N, F, HO, WO] = conv_3[N, F, HO, WO].call(`max`, [0.0]);
  conv_4[N, F, HO, WO] +=! conv_3_relu[N, C, -1 + HO + KH, -1 + WO + KW].when([-1 + HO + KH >= 0, -1 + HO + KH < 13, -1 + WO + KW >= 0, -1 + WO + KW < 13], 0.0) * const_4_[KH, KW, C, F] where HO in 13, WO in 13;
  mpool_2[N, C, HO, WO] >=! conv_4[N, C, HO * 2 + KH, WO * 2 + KW].call(`max`, [0.0]) where HO in 6, WO in 6, KH in 3, KW in 3;
  reshape_0[N0, N1] = mpool_2[N0, N1 // 36 % 256, N1 // 6 % 6, N1 % 6] where N1 in 9216;
  dense_0[N, M] +=! reshape_0[N, K] * const_5_[K, M];
  dense_0_relu[N, M] = dense_0[N, M].call(`max`, [0.0]);
  dense_1[N, M] +=! dense_0_relu[N, K] * const_6_[K, M];
  dense_1_relu[N, M] = dense_1[N, M].call(`max`, [0.0]);
  dense_2[N, M] +=! dense_1_relu[N, K] * const_7_[K, M];
''', feed_dict=feed_dict).emit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  print('Result from Antares = %s' % sess.run([output_logits]))
  print('Result from Tensorflow = %s' % sess.run([output_logits_tf]))

