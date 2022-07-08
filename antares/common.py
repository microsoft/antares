# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import math

class Mock(object):
  pass

backend = os.environ.get('BACKEND', None)
AntaresGlobal = Mock()

def product(arrlist):
  result = 1
  for x in arrlist:
    result *= int(x)
  return result

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

def local_get_dir_file(rel_file, dir_sid=None, prefix=None):
  if dir_sid is None:
    dir_sid = os.environ['DIR_SID'] if 'DIR_SID' in os.environ else '_'
  driver_path = os.environ['ANTARES_DRIVER_PATH']
  if prefix is not None:
    driver_path = os.path.join(driver_path, prefix)
  dir_space = os.path.join(driver_path, 'cache')
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

def parse_defination(code, key, defs):
  import re
  if re.search(f'\\b{key}\\b', code):
    return [defs]
  return []

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
    for val in init_vals:
      prod_rest = target_size
      for i in range(1, len(val)):
        val[i] = math.gcd(prod_rest, val[i])
        prod_rest //= val[i]
      val[0] = prod_rest
    unique_vals = []
    for item in init_vals:
      if item not in unique_vals:
        unique_vals.append(item)
    init_vals = unique_vals
    del unique_vals
    self._config[key] = {'_type': 'factor', '_value': [target_size, num_outputs], '_init': init_vals}
    if self._candidate:
      return self._candidate[key]
    return [-1] + init_vals[0][1:]

  def define_reorder(self, key, count, policy='all', init_vals=[]):
    if not init_vals:
      init_vals = [[x for x in range(count)]]
    assert isinstance(count, int), "Reorder value must be integer type."
    assert policy == 'all', "Unhandled reorder policy: %s" % policy
    self._config[key] = {'_type': 'perm', '_value': count, '_init': init_vals}
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
    return list(reversed(slices))

use_format = 'FORMAT' in os.environ

def cpp_format(code):
  if not use_format:
    return code
  import tempfile
  ftemp = tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), suffix='.cpp')
  with open(ftemp.name, 'w') as fp:
    fp.write(code)
  st, output = subprocess.getstatusoutput('clang-format < ' + ftemp.name)
  if st != 0:
    return code
  return output.replace(' @\n', '@\n')
