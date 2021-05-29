#!/usr/bin/env python3

# mpiexec -n 2 --allow-run-as-root --map-by slot --bind-to none -x N=$((1024 * 1024)) -x R=1 -x OP='all_reduce:+' ./tf_nccl_benchmark.py
import os, tensorflow as tf
from tensorflow.contrib import antares

if tf.version.VERSION.startswith('2.'):
  tf = tf.compat.v1
  tf.disable_eager_execution()

rank, size, local_rank = antares.init_communicate_config(expect_nodes=2)
count = int(os.environ.get('N', '4096'))
op = os.environ.get('OP', 'all_reduce:+')
repeat = int(os.environ.get('RP', '1'))

if not op.startswith('all_gather:'):
  count *= size

input_0 = tf.get_variable('input_0', count, 'float32', initializer=tf.initializers.ones('float32'))
[input_0] = antares.metric([input_0])
for i in range(repeat): [input_0] = antares.communicate(op, [input_0], names=["input_0"])
[output_0] = antares.metric([input_0])

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.visible_device_list = str(local_rank)

with tf.Session(config=tf_config) as sess:
  sess.run(tf.global_variables_initializer())
  print("Node[%d/%d]: Tensor output =" % (rank, size), sess.run(output_0))
  for x in range(4):
    sess.run(output_0)
  print("Node[%d/%d]: Tensor output properties: shape = %s, dtype = %s" % (rank, size, output_0.shape, output_0.dtype))
