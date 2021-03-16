# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import tvm
from tvm import auto_scheduler
from antares.common import local_get_dir_file, AntaresGlobal

GLOBAL_TUNER = None

@tvm._ffi.register_func("auto_scheduler.local_builder.build", override=True)
def local_builder_build(inputs, timeout, n_parallel, build_func="default", verbose=1):
  build_results = []
  for i in range(len(inputs)):
    cfg_path = local_get_dir_file('my_kernel.cfg', i)
    try:
      os.remove(cfg_path)
    except:
      pass
    tvm.auto_scheduler.measure_record.save_records(cfg_path, [inputs[i]], [tvm.auto_scheduler.measure.MeasureResult([0.0], 0, 0, 0, 0)])
    build_results.append(tvm.auto_scheduler.measure.BuildResult(cfg_path, (), 0, 0, 0))
  return build_results

@tvm._ffi.register_func("auto_scheduler.rpc_runner.run", override=True)
def local_run(inputs, build_results, key, host, port, priority=1, n_parallel=1, timeout=10, number=3, repeat=1, min_repeat_ms=0, cooldown_interval=0.0, enable_cpu_cache_flush=False, verbose=1):
  global GLOBAL_TUNER
  assert GLOBAL_TUNER is not None
  tuner = GLOBAL_TUNER

  measure_inputs = []
  for i in range(len(build_results)):
    with open(str(build_results[i].filename), 'r') as fp:
      line = fp.readline()
    jobj = json.loads(line)
    config = jobj['i'][1]
    measure_inputs.append(tvm.autotvm.measure.MeasureInput(tuner.task.target, tuner.task, json.dumps(config)))

  results = GLOBAL_TUNER.measure_batch(measure_inputs)
  measure_results = []
  for i in range(len(results)):
    measure_results.append(tvm.auto_scheduler.measure.MeasureResult(results[i].costs, results[i].error_no, '', results[i].all_cost, results[i].timestamp))
  return measure_results

@auto_scheduler.register_workload
def auto_template():
  _, arg_bufs = AntaresGlobal.default_tune_op.get_template_op()
  return arg_bufs

def create_auto_task(tvm_target):
  return auto_scheduler.create_task(auto_template, (), tvm_target)

class MainTuner(object):

  def __init__(self, task, **kwargs):
    self.task = task
    self.measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    self.auto_task = create_auto_task(task.target)

  def cleanup(self):
    del self.measure_ctx

  def tune(self, n_trial, **kwargs):
    global GLOBAL_TUNER
    GLOBAL_TUNER = self
    try:
      auto_scheduler.auto_schedule(self.auto_task, tuning_options=auto_scheduler.TuningOptions(num_measure_trials=n_trial, num_measures_per_round=self.task.n_parallel, runner=self.measure_ctx.runner, measure_callbacks=[]))
    except:
      import traceback
      traceback.print_exc()
      exit(1)
