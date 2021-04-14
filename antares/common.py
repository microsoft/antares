# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess


class Mock(object):
  pass

backend = os.environ['BACKEND']
AntaresGlobal = Mock()

def wait_for(func, timeout=None, args=[]):
  if not timeout:
    return func(*args)
  def timeout_handler():
    raise Exception("Error: Timeout during Kernel warmup")
  from threading import Timer
  my_timer = Timer(timeout, timeout_handler, [])
  my_timer.start()
  res = func(*args)
  my_timer.cancel()
  del my_timer
  return res

def local_get_dir_file(rel_file, dir_sid=None):
  if dir_sid is None:
    dir_sid = os.environ['DIR_SID'] if 'DIR_SID' in os.environ else '_'
  dir_space = os.path.join(os.environ['ANTARES_DRIVER_PATH'], 'cache')
  os.system('mkdir -p "%s/%s"' % (dir_space, dir_sid))
  return "%s/%s/%s" % (dir_space, dir_sid, rel_file)

def run_process_with_timeout(args, timeout=None, envs=None):
  try:
    if timeout is not None:
      args = ['/usr/bin/timeout', str(timeout)] + args
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=envs)
    retcode = proc.wait()
    return retcode == 0
  except subprocess.TimeoutExpired:
    print('Timed out - killing', proc.pid)
    proc.kill()
    return False

def get_type_size(dtype):
  for i in reversed(range(len(dtype))):
    if not dtype[i].isdigit():
      bits = int(dtype[i + 1:])
      assert bits % 8 == 0, "Data type size must align with 8-bit byte size."
      return bits // 8
  raise Exception("Unrecognized data size for data type: %s" % dtype)

class AutoConfig(object):

  def __init__(self):
    self._config = dict()
    self._candidate = None

  def get_config_space(self):
    return self._config

  def set_candidate(self, candidate):
    self._candidate = candidate

  def define_split(self, key, target_size, num_outputs, init_vals=[]):
    assert isinstance(target_size, int), "Split target must be integer type."
    if not init_vals:
      init_vals = [[-1] + [1] * (num_outputs - 1), ]
    else:
      unique_vals = []
      for item in init_vals:
        if item not in unique_vals:
          unique_vals.append(item)
      init_vals = unique_vals
    self._config[key] = {'_type': 'factor', '_value': [target_size, num_outputs], '_init': init_vals}
    if self._candidate:
      return self._candidate[key]
    return [-1] + init_vals[0][1:]

  def define_reorder(self, key, count, policy='all', init_vals=[]):
    if not init_vals:
      init_vals = [[x for x in range(count)]]
    assert isinstance(count, int), "Reorder value must be integer type."
    assert policy == 'all', "Unhandled reorder policy: %s" % policy
    self._config[key] = {'_type': 'perm', '_value': count}
    if self._candidate:
      return self._candidate[key]
    return init_vals[0]

  def define_knob(self, key, choices, init_vals=[]):
    if not init_vals:
      init_vals = [0]
    self._config[key] = {'_type': 'choice', '_value': [x for x in range(len(choices))], '_init': init_vals}
    if self._candidate:
      return choices[self._candidate[key]]
    return choices[init_vals[0]]

  def apply_split(self, s, output, ax, sizes):
    slices = []
    for sz in reversed(sizes[1:]):
      ax, ai = s[output].split(ax, factor=sz)
      slices.append(ai)
    slices.append(ax)
    return reversed(slices)
