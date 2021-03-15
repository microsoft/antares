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

B, S, N, H, I = 6, 128, 12, 48, 1024

input_tensor = tf.get_variable('input_tensor', [B, S, N, H], tf.float32, initializer=tf.initializers.ones(tf.float32), trainable=False)

layer_output_norm = antares.make_op(ir=f'''
  merged_layer_local[R, B, S1, N1, H1] +=! input_tensor[B, S1, N, H] * qkv_weight[R, N, H, N1, H1];
  merged_layer_trans[R, B, N1, S1, H1] = merged_layer_local[R, B, S1, N1, H1] + qkv_bias[R, N1, H1];
  attention_scores[B, N1, S1, S2] +=! merged_layer_trans[0, B, N1, S1, H1] * merged_layer_trans[1, B, N1, S2, H1] / const({H}).cast(`float32`);
    softmax_1_temp0[B, N1] >=! attention_scores[B, N1, S1, S2];
    softmax_1_temp1[B, N1] +=! (attention_scores[B, N1, S1, S2] - softmax_1_temp0[B, N1]).call(`exp`);
  attention_probs[B, N1, S1, S2] = (attention_scores[B, N1, S1, S2] - softmax_1_temp0[B, N1]).call(`exp`) / softmax_1_temp1[B, N1];
  context_layer_trans[B, S1, N1, H1] +=! attention_probs[B, N1, S1, S2] * merged_layer_trans[2, B, N1, S2, H1];
  attention_local[B, S1, N2, H2] +=! context_layer_trans[B, S1, N1, H1] * attention_weight[N1, H1, N2, H2];
  attention_output[B, S1, N2, H2] = attention_local[B, S1, N2, H2] + attention_bias[N2, H2];
  layer_norm_1_src[B, S1, N2, H2] = attention_output[B, S1, N2, H2] + input_tensor[B, S1, N2, H2];
    layer_norm_1_temp0[B, S1] += layer_norm_1_src[B, S1, N2, H2];
    layer_norm_1_temp1[B, S1] += layer_norm_1_src[B, S1, N2, H2] * layer_norm_1_src[B, S1, N2, H2];
  attention_output_norm[B, S1, N2, H2] = (layer_norm_1_src[B, S1, N2, H2] * {N * H} - layer_norm_1_temp0[B, S1]) * (layer_norm_1_temp0[B, S1] * {N * H} - layer_norm_1_temp1[B, S1] * layer_norm_1_temp1[B, S1]).call(`max`, [1e-8]).call(`rsqrt`);
  intermediate_local[B, S1, I] +=! attention_output_norm[B, S1, N2, H2] * intermediate_weight[N2, H2, I];
  intermediate[B, S1, I] = intermediate_local[B, S1, I] + intermediate_bias[I];
  intermediate_gelu[B, S1, I] = 0.5 * (1.0 + (0.79788456 * (intermediate[B, S1, I] + 0.044715 * intermediate[B, S1, I] * intermediate[B, S1, I] * intermediate[B, S1, I])).call(`tanh`));
  layer_output_local[B, S1, N2, H2] +=! intermediate_gelu[B, S1, I] * output_weight[I, N2, H2];
  layer_output[B, S1, N2, H2] = layer_output_local[B, S1, N2, H2] + output_bias[N2, H2];
  layer_norm_2_src[B, S1, N2, H2] = layer_output[B, S1, N2, H2] + attention_output_norm[B, S1, N2, H2];
    layer_norm_2_temp0[B, S1] += layer_norm_2_src[B, S1, N2, H2];
    layer_norm_2_temp1[B, S1] += layer_norm_2_src[B, S1, N2, H2] * layer_norm_2_src[B, S1, N2, H2];
  layer_output_norm[B, S1, N2, H2] = (layer_norm_2_src[B, S1, N2, H2] * {N * H} - layer_norm_2_temp0[B, S1]) * (layer_norm_2_temp0[B, S1] * {N * H} - layer_norm_2_temp1[B, S1] * layer_norm_2_temp1[B, S1]).call(`max`, [1e-8]).call(`rsqrt`);
''', feed_dict={
  'input_tensor': input_tensor,
  'qkv_weight': create_param('qkv_weight', [3, N, H, N, H]),
  'qkv_bias': create_param('qkv_bias', [3, N, H]),
  'attention_weight': create_param('attention_weight', [N, H, N, H]),
  'attention_bias': create_param('attention_bias', [N, H]),
  'intermediate_weight': create_param('intermediate_weight', [N, H, I]),
  'intermediate_bias': create_param('intermediate_bias', [I]),
  'output_weight': create_param('output_weight', [I, N, H]),
  'output_bias': create_param('output_bias', [N, H]),
}).emit()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  print('Result = %s' % sess.run([layer_output_norm]))

