# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys, os
import urllib.request

from tvm.autotvm.tuner import XGBTuner
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob

def online_download(url_path, file_dest):
  req = urllib.request.Request(url_path, headers={}, method='GET')
  with urllib.request.urlopen(req) as fp:
    file_content = fp.read().decode()
  with open(file_dest, 'w') as fp:
    fp.write(file_content)

class MainTuner(XGBTuner):

  def __init__(self, task, **kwargs):
    if 'plan_size' not in kwargs:
      kwargs['plan_size'] = 32
    super(MainTuner, self).__init__(task, **kwargs)

    try:
      from tvm.autotvm.tuner.adaptive_sampler import AdaptiveSampler
    except:
      import sys, os
      dir_path = os.path.dirname(os.path.abspath(sys.modules[XGBTuner.__module__].__file__))
      print('Merging Chameleon plugins from `https://bitbucket.org/act-lab/chameleon` into `%s`..' % dir_path)
      online_download('https://bitbucket.org/act-lab/chameleon/raw/1699894b6f8cec37af4fef0b90fd6e6441f584ae/chameleon/tuner/sampler.py', os.path.join(dir_path, 'sampler.py'))
      online_download('https://bitbucket.org/act-lab/chameleon/raw/1699894b6f8cec37af4fef0b90fd6e6441f584ae/chameleon/tuner/adaptive_sampler.py', os.path.join(dir_path, 'adaptive_sampler.py'))
      from tvm.autotvm.tuner.adaptive_sampler import AdaptiveSampler
    finally:
      self.sampler = AdaptiveSampler(self.task, self.plan_size)
      self.next_update = self.plan_size

  def update(self, inputs, results):
    # Update results without fitting model
    self.train_ct = len(self.xs)
    super().update(inputs, results)
    if len(self.visited) >= self.next_update:
      # Fit model with adaptive sampler
      self.train_ct = -1
      super().update([], [])
      assert(self.train_ct == 0)

      maximums = self.trials
      print("  >> Adaptive Sampling of %d samples.." % len(maximums))
      samples = [point2knob(config, self.dims) for config in maximums]
      reduced_samples = self.sampler.sample(samples, self.dims)
      maximums = [knob2point(sample, self.dims) for sample in reduced_samples]
      
      print("  >> Adaptive Sampling: Reducing samples to %d" % len(maximums))
      self.trails = maximums
      self.next_update += len(maximums)

