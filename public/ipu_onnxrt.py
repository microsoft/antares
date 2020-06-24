#!/usr/bin/python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import popart
import numpy as np
import os, sys
import onnxruntime

model_path = sys.argv[1]

import onnxruntime
sess = onnxruntime.InferenceSession(model_path)

space_input, space_output = {}, {}
for it in sess.get_inputs():
  space_input[it.name] = np.array([1.0] * np.product(it.shape), dtype=np.float32)
for it in sess.get_outputs():
  space_output[it.name] = popart.AnchorReturnType("ALL")

if 'PROF' in os.environ:
  popart.getLogger().setLevel("DEBUG")

anchors = space_output

dataFeed = popart.DataFlow(1, anchors)

try:
 session = popart.InferenceSession(model_path, dataFeed, popart.DeviceManager().acquireAvailableDevice())
 print('Using IPU Hardware ..')
except:
 session = popart.InferenceSession(model_path, dataFeed, popart.DeviceManager().createIpuModelDevice({}))
 print('Using IPU Model ..')

session.prepareDevice()

anchors = session.initAnchorArrays()
stepio = popart.PyStepIO(space_input, anchors)

session.run(stepio)
import time
t1 = time.time()
step = 100
for i in range(step):
  session.run(stepio)
t2 = time.time()

print("=>", anchors)
print('Time:', (t2 - t1) / step)
