# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from tvm.autotvm.tuner import XGBTuner

class MainTuner(XGBTuner):

  def __init__(self, task, **kwargs):
    if 'plan_size' not in kwargs:
      kwargs['plan_size'] = 32
    if 'feature_type' not in kwargs:
      kwargs['feature_type'] = 'knob'
    if 'num_threads' not in kwargs:
      kwargs['num_threads'] = 4
    super(MainTuner, self).__init__(task, **kwargs)

  def cleanup(self):
    try:
      self.cost_model._close_pool()
    except:
      pass
