# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys, time, subprocess, os
import random
import hashlib
import traceback
import numpy as np
import logging
import math
import re
import json
import importlib
import signal
import collections

import tvm
from tvm import autotvm
from tvm.autotvm.task.dispatcher import ApplyConfig
from tvm.autotvm.task import ConfigEntity

from antares.common import *
from templates.auto.generic import custom_dtypes

signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(1))

tvm_target = 'cuda'
eval_program_timeout = 30
krnl_compile_timeout = 20
verbose = int(os.environ.get('VERBOSE', '1'))

try:
  platform_config = importlib.import_module('platforms.%s.config' % backend)
except ModuleNotFoundError:
  raise Exception('>> Platform config for backend %s not found' % backend)
except:
  traceback.print_exc()
  exit(1)

def get_search_space(config_space):
  search_space = {}
  for _, name in enumerate(config_space.space_map):
    curr = config_space.space_map[name]
    if (curr.__class__ == tvm.autotvm.task.space.SplitSpace):
      search_space[name] = {"_type": "factor", "_value": [curr.product, curr.num_output]}
    elif (curr.__class__ == tvm.autotvm.task.space.OtherOptionSpace):
      search_space[name] = {"_type": "choice", "_value": [x.val for x in curr.entities]}
    elif (curr.__class__ == tvm.autotvm.task.space.ReorderSpace):
      search_space[name] = {"_type": "perm", "_value": curr.num_output}
    else:
      raise Exception("Cannot recognize search space type: %s" % (config_space.space_map[name].__class__))
  return search_space

def translate_code(code):
  assert(len(code.split('extern "C"')) == 2)
  def get_kernel_metadata():
    inp_args, outp_args = [], []

    global_arg_bufs = os.environ.get('GLOBAL_ARG_PROPS', '')
    if not global_arg_bufs:
      global_arg_bufs = AntaresGlobal.current_arg_bufs
    else:
      global_arg_bufs = json.loads(global_arg_bufs)

    for buf in global_arg_bufs['_in']:
      if buf['name'].startswith('_'):
        # Just for Auto Shard
        assert(buf['dtype'] == 'int32' and buf['shape'] == [1])
        continue
      inp_args.append('-'.join([str(x) for x in buf['shape']]) + '/' + buf['dtype'] + '/' + buf['name'])
    for buf in global_arg_bufs['_out']:
      outp_args.append('-'.join([str(x) for x in buf['shape']]) + '/' + buf['dtype'] + '/' + buf['name'])

    header_meta = '///' + ','.join(inp_args) + ':' + ','.join(outp_args) + '\n// BACKEND = %s\n' % backend
    properties = "// CONFIG: %s\n// COMPUTE_V1: %s\n" % (os.environ['CONFIG'].strip(), os.environ['COMPUTE_V1'] if os.environ['OP'] == 'auto.generic' else os.environ['OP'])
    return header_meta + properties

  code = platform_config.do_native_translation(code, attrs=AntaresGlobal.attrs)
  try:
    defs = platform_config.get_intrisic_defs() + '\n'
  except:
    defs = ''
  return '%s\n%s%s' % (get_kernel_metadata(), defs, code)

def device_properties():
  props = tvm.runtime.ndarray.gpu(0)
  with open('%s/device_properties.cfg' % os.environ['ANTARES_DRIVER_PATH'], 'r') as fp:
    mem_bandwith = []
    while True:
      line = fp.readline()
      if not line:
        break
      key, val = line.split(': ')
      if key in ('GlobalMemoryBusWidth', 'MemoryClockRate'):
        mem_bandwith.append(float(val))
    mem_bandwith = 'inf' if not mem_bandwith else np.product(mem_bandwith) * 2.5e-7
    props.mem_bandwith = float(mem_bandwith)
  return props

def compile_source(code):
  if os.environ.get('HTTP_SERVICE', ''):
    return bytearray()
  kernel_src = local_get_dir_file("my_kernel.cc")
  kernel_out = local_get_dir_file("my_kernel.out")
  with open(kernel_src, 'w') as fp:
    fp.write(translate_code(code))
  args = platform_config.get_compile_kernel_args(kernel_src, kernel_out, device_properties())

  if verbose:
    print('[Build (pid=%d)]' % os.getpid(), ' '.join(args))
  assert run_process_with_timeout(args, krnl_compile_timeout), "Compilation failed for: Bad kernel code, or Time limit exceeded?\nFailure command: %s\n" % ' '.join(args)
  with open(kernel_out, "rb") as fp:
    return bytearray(fp.read())

def compute_gflops(flop, t):
  try:
    return flop / (t * 1e3) / 1e6
  except:
    return 0.0

def codehub_db(compute_key, source_code=None):
  compute_key = compute_key.split('##')[0].strip()
  digest = hashlib.sha256(compute_key.encode()).hexdigest()
  os.system('mkdir -p ./codehub')
  code_path = './codehub/%s.%s' % (digest, backend)
  if not source_code:
    if os.path.exists(code_path):
      print('  >> Codehub Key = %s.%s' % (digest, backend))
      with open(code_path, 'r') as fp:
        code = fp.read()
      return code
    else:
      return None
  else:
    with open(code_path, 'w') as fp:
      fp.write(source_code)
    return code_path

def get_target_source(s, arg_bufs, dir_sid=None):
  if s is not None:
      lower_source = str(tvm.lower(s, arg_bufs, simple_mode=True))

      lower_file = local_get_dir_file('my_kernel.lower', dir_sid=dir_sid)
      with open(lower_file, 'w') as fp:
        fp.write(lower_source)

      # Verify Lower Code Code
      if len(('\n' + lower_source).split('\nprimfn(')) != 2:
        raise Exception('[Not Support Multi Unfuse-able kernels]\n\n' + lower_source)

      max_threads_per_block = device_properties().max_threads_per_block
      max_shared_memory_per_block = device_properties().max_shared_memory_per_block
      assert max_threads_per_block > 0 and max_shared_memory_per_block >= 0, '[Error] Invalid device properties, maybe device is not detected correctly.'

      lower_lines = lower_source.split('\n')
      thread_extents, allocate_shared = [], []
      for ll in lower_lines:
        if ll.strip().startswith('attr [IterVar(') and ll.find(' "thread_extent" = ') >= 0:
          thread_name = ll.split('attr [IterVar(')[-1].split(':')[0]
          thread_val = int(ll.split(' "thread_extent" = ')[-1].split(';')[0].strip().split(' ')[0])
          thread_extents.append((thread_name, thread_val))
        elif ll.strip().startswith('allocate(') and ll.find('.shared, ') >= 0 and ll.endswith(");"):
          parts = ll[:-2].split(', ')[1:]
          allocate_type = parts[0]
          allocate_val = int(np.product(eval(parts[1])))
          allocate_shared.append((allocate_type, allocate_val))

      reserved_axes = dict()
      for thread_name, thread_val in thread_extents:
        if thread_name in reserved_axes:
          assert reserved_axes[thread_name] == thread_val, "Invalid code: Multiple hints for thread extent conflict with each other: %d v.s. %d" % (reserved_axes[thread_name], thread_val)
        else:
          reserved_axes[thread_name] = thread_val

      num_threads = 1
      for thread_name in ['threadIdx.x', 'threadIdx.y', 'threadIdx.z']:
        num_threads *= reserved_axes.get(thread_name, 1)
      assert num_threads <= max_threads_per_block, "Invalid kernel code: using num_threads(%d) > max_threads_per_block(%d)" % (num_threads, max_threads_per_block)

      shared_memory_in_bytes = 0
      for allocate_type, allocate_size in allocate_shared:
        if allocate_type.startswith('custom['):
          type_name = allocate_type[7:].split(']')[0]
          shared_memory_inc = int(custom_dtypes[type_name][-1].split('@')[-1])
        else:
          shared_memory_inc = 8 * np.dtype(allocate_type).itemsize
        assert shared_memory_inc % 8 == 0, "The bits of shared_memory is not aligned with 8-bit bytes."
        shared_memory_in_bytes += shared_memory_inc // 8 * allocate_size

      if shared_memory_in_bytes > max_shared_memory_per_block:
        raise Exception("Invalid kernel code: using shared_memory_in_bytes %d > max_shared_memory_per_block %d" % (shared_memory_in_bytes, max_shared_memory_per_block))

      # Compile Source Code
      def build_template():
        return tvm.build(s, arg_bufs, tvm_target, name='template_op')
      func = build_template()

  assert(len(func.imported_modules) == 1)
  device_source = translate_code(func.imported_modules[0].get_source())
  kernel_path = local_get_dir_file('my_kernel.cc', dir_sid=dir_sid)
  return device_source, kernel_path

def evaluate_perf(kernel_path, task_flop, dev_id):
  try:
    eval_client = importlib.import_module('platforms.%s.evaluator.client' % backend)
  except ModuleNotFoundError:
    print('>> Evaluator for backend %s not found, skipping evaluation.' % backend)
    return None
  except:
    traceback.print_exc()
    return None

  def handle_result(result):
    if verbose:
      print('\n[EvalAgent] Results =', json.dumps(result))
    if 'RESULT' in os.environ:
      if abs(float(os.environ['RESULT']) / result['K/0'] - 1.0) > 1e-6:
        result['TPR'] = None

    t = result.get('TPR', None)
    if t is None:
      print("\n[Antares] Incorrect compute kernel from evaluator.")
    else:
      gflops = compute_gflops(task_flop, t)
      if verbose:
        print("\n[Antares] Average time cost / run = %g sec, %g gflops." % (t, gflops))
      with open(local_get_dir_file('result.txt'), 'w') as fp:
        fp.write(str(t) + '\n')
        if 'K/0' in result:
          fp.write(str(result['K/0']) + '\n')
    if os.environ['OP'] == 'auto.generic' and os.environ.get('COMMIT', ''):
      kernel_path = codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + '\n// Saved Perf = %g sec / run' % t)
      print('  >> Update current code to codehub: %s' % kernel_path)

  def do_evaluate():
    try:
      expected_timeout = os.environ.get('EXPECTED_TIMEOUT', '')
      if expected_timeout in ('', 'inf'):
        expected_timeout = ''
      else:
        expected_timeout = float(expected_timeout)
        expected_timeout = max(expected_timeout * 1.1, expected_timeout + 0.1)

      results = eval_client.eval(kernel_path=local_get_dir_file('my_kernel.cc'),
                  expected_timeout=expected_timeout,
                  dev_id=dev_id,
                )
      return results
    except:
      traceback.print_exc()
      return None

  exec_fd, _ = system_lock([dev_id])
  results = wait_for(do_evaluate, eval_program_timeout if int(os.environ.get('STEP', '0')) > 0 else None)
  if results is not None:
    handle_result(results)
  exec_fd()
  return results

def compute_mem_ratio(tpr):
  access_bytes = int(os.environ.get('MEM_ACCESS', '0'))
  ratio = np.ceil(access_bytes * 1e-7 / tpr / device_properties().mem_bandwith)
  return min(int(ratio), 100)

def run_config_entity(params_given, dir_sid, expected_timecost='inf', tune_dev_id=0):
  dir_sid = str(dir_sid)
  result_file = local_get_dir_file('result.txt', dir_sid)
  try:
    os.remove(result_file)
  except:
    pass
  config_str = json.dumps(params_given)
  envs = {}
  envs['CONFIG'] = config_str
  envs['DIR_SID'] = dir_sid
  envs['DEV_KEY'] = str(tune_dev_id)
  expected_timecost = float(expected_timecost)
  if math.isinf(expected_timecost):
    envs['EXPECTED_TIMEOUT'] = ''
  else:
    envs['EXPECTED_TIMEOUT'] = str(expected_timecost)
  env_str = ' '.join(["%s='%s'" % (x, envs[x]) for x in envs])
  print("  >> [ ] Param_entity on sid = %s: config = '%s', dev_id = %d, upper_bound_tpr = %.6e s" % (dir_sid, config_str, tune_dev_id, expected_timecost))
  log_path = local_get_dir_file('evaluator.log', dir_sid)
  exe_cmd = "%s python%d %s > %s.stdout 2> %s.stderr" % (env_str, sys.version_info.major, ' '.join(sys.argv), log_path, log_path)
  try:
    assert 0 == os.system(exe_cmd)
    with open(result_file, 'r') as fp:
      parts = fp.read().split()
      result = float(parts[0].strip())
      digest = float(parts[1].strip()) if len(parts) > 1 else float('inf')
  except:
    result = digest = float('inf')
  print("  >> [*] Param_entity on sid = %s: config = '%s', tpr = `%.6f`, digest = `%g`, mem_occupy = %d %%" % (dir_sid, config_str, result, digest, compute_mem_ratio(result)))
  return result


def main_compute(code_only=False):
  tvm.register_func('tvm_callback_cuda_compile', compile_source, override=True)
  logging.getLogger('autotvm').setLevel(logging.DEBUG)
  logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

  default_tune_op = importlib.import_module('templates.' + (os.environ['OP'] if 'OP' in os.environ else 'auto.generic'))
  if verbose:
    print('  >> Backend = %s, Python PID = %s, Task = %s;' % (backend, os.getpid(), default_tune_op.__name__))

  task = autotvm.task.create("template_op", args=(), target=tvm_target)

  def json_to_config(json_dict, index=-1, code_hash=None):
    if not isinstance(json_dict, list):
      json_list = []
      for key in json_dict:
        json_list.append([key, 'ot' if type(json_dict[key]) is not list else ('sp' if json_dict[key][0:1] == [-1] else 're'), json_dict[key]])
      json_dict = json_list
    config = ConfigEntity.from_json_dict({"index": index, "time": "", "code_hash": code_hash, "entity": json_dict})
    # config = ConfigEntity.from_json_dict({"i": index, "t": "", "c": code_hash, "e": json_dict})
    return config

  def config_to_json(config):
    if config is None:
      return {}
    if isinstance(config, str):
      return json.loads(config)
    jobj = config.to_json_dict()['entity']
    # jobj = config.to_json_dict()['e']
    json_dict = dict()
    for i in range(len(jobj)):
      assert(jobj[i][1] in ['sp', 'ot', 're'])
      json_dict[jobj[i][0]] = jobj[i][2]
    return json_dict

  num_trials = int(os.environ['STEP']) if 'STEP' in os.environ else 0

  config = os.environ.get('CONFIG', '').strip()
  if config != '':
    best_config = config
  elif 'NNI_TRIAL_JOB_ID' in os.environ:
    if os.environ['NNI_TRIAL_JOB_ID'] == '@':
      search_space = get_search_space(task.config_space)
      json_space = json.dumps(search_space)
      dump_to_file='./search_space.json'
      print("\n>> Writing Search Space to '%s', Search Space = %s;" % (dump_to_file, json_space))
      with open("search_space.json", "w") as fp:
        fp.write(json_space)
      sys.exit(0)

    try:
      import nni
      params_given = nni.get_next_parameter()
      if params_given is None:
        raise
      local_dir_id = os.environ['NNI_TRIAL_JOB_ID']
    except:
      params_given = default_tune_op.get_choice_example()
      local_dir_id = '_'
    t = run_config_entity(params_given, local_dir_id)
    gflops = compute_gflops(task.flop, t)
    print('[Antares-engine] Final entity result is: %g' % gflops)
    try:
      nni.report_final_result(gflops)
    except:
      print('[Antares-engine] (not reporting final result to NNI.)')
    exit(0)

  elif num_trials > 0:
    dev_num = platform_config.get_execution_parallism()
    if dev_num <= 0:
        raise Exception("No valid device found for backend: %s." % backend)
    batch_size = int(os.environ.get('BATCH', '16'))

    from concurrent.futures import ThreadPoolExecutor
    try:
      if platform_config.allow_concurrent_compile_execution():
        raise Exception()
      worker_size = 1
    except:
      worker_size = batch_size
    thread_pool = ThreadPoolExecutor(max_workers=worker_size)

    task.antares_helper = Mock()
    task.antares_helper.json_to_config = json_to_config
    task.antares_helper.config_to_json = config_to_json
    task.antares_helper.to_json_search_space = get_search_space

    tuner_type = os.environ.get('TUNER', '')
    if not tuner_type:
      comp = os.environ['COMPUTE_V1']
      if '=!' in comp and 'plan/' not in comp[comp.find(' ##') + 1:] and ';' not in comp and backend in ['c-rocm', 'c-cuda', 'c-hlsl', 'c-ocl']:
        tuner_type = 'AutoTVM2'
      else:
        tuner_type = 'XGBoost'
    print('  >> MAKE_PARA = %d/%d, EXEC_PARA = %d, TUNER = %s' % (worker_size, batch_size, dev_num, tuner_type))

    auto_commit = os.environ.get('COMMIT', '')
    if auto_commit:
      saved_code = codehub_db(os.environ['COMPUTE_V1'])
      if saved_code is not None and auto_commit != 'force':
        raise Exception("Saved code has existed in codehub. Please try COMMIT=force to override it.")
      os.environ.pop('COMMIT')

    try:
      tuner = importlib.import_module('tuner.%s.main' % tuner_type)
      tuner = tuner.MainTuner(task)
    except:
      raise Exception('>> Cannot import Antares Tuner: %s' % tuner_type)

    if tuner is not None:
      AntaresGlobal.current_step = 0

      def measure_batch(inputs):
        results, futures = [], []
        best_slot = -1
        expected_timecost = tuner.task.best.timecost
        for i in range(len(inputs)):
          futures.append(thread_pool.submit(run_config_entity, config_to_json(inputs[i].config), AntaresGlobal.current_step + i + 1, expected_timecost, i % dev_num))
        for i in range(len(inputs)):
          t = futures[i].result()
          if t < tuner.task.best.timecost:
            best_slot = AntaresGlobal.current_step + i + 1
            tuner.task.best.timecost = t
            tuner.task.best.config = inputs[i].config
            tuner.task.best.occur = best_slot
          results.append(autotvm.measure.MeasureResult(costs=(t,), error_no=0, all_cost=i, timestamp=time.time()))
        AntaresGlobal.current_step += len(results)

        print('\nSTEP[%d / %d] Current Best Config = %s, Perf = %g Gflops, MemRatio = %g %%, Occur Step = %d;' % (
          AntaresGlobal.current_step,
          num_trials,
          json.dumps(config_to_json(tuner.task.best.config)),
          compute_gflops(tuner.task.flop, tuner.task.best.timecost),
          compute_mem_ratio(tuner.task.best.timecost),
          tuner.task.best.occur))

        if auto_commit and best_slot >= 0:
          with open(local_get_dir_file('my_kernel.cc', best_slot), 'r') as fp:
            device_source = fp.read()
          with open(local_get_dir_file('result.txt', best_slot), 'r') as fp:
            t = float(fp.read().split()[0])
          kernel_path = codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + '\n// Saved Perf = %g sec / run; Step Produced = %d; Planned Steps = %d;' % (t, best_slot, num_trials))
          print('  >> Update current code to codehub: %s' % kernel_path)
        return results

      tuner.task.best = Mock()
      tuner.task.best.timecost = float('inf')
      tuner.task.best.config = None
      tuner.task.best.occur = -1

      tuner.measure_batch = measure_batch
      tuner.measure_batch.n_parallel = batch_size
      callbacks = []

      history_log_for_transfer_learning = os.environ.get('RECORD', '')

      if history_log_for_transfer_learning:
        callbacks.append(autotvm.callback.log_to_file(history_log_for_transfer_learning))
        # Enable Transfer Learning for Incremental Task
        if os.path.exists(history_log_for_transfer_learning):
          print('  >>  Loading incremental history from log file: %s ..' % history_log_for_transfer_learning)
          tuner.load_history(autotvm.record.load_from_file(history_log_for_transfer_learning))

      tuner.tune(n_trial=num_trials, measure_option=autotvm.measure_option(
          builder=autotvm.LocalBuilder(n_parallel=batch_size),
          runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
      ), callbacks=callbacks)
      assert not math.isinf(tuner.task.best.timecost), "Not valid config found in the whole tuning."
      best_config = json.dumps(config_to_json(tuner.task.best.config))

      if auto_commit:
          device_source = codehub_db(os.environ['COMPUTE_V1'])
          codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + '\n// Antares Tuning Completed in %d steps.' % AntaresGlobal.current_step)

      print("\n[Best Config] CONFIG='%s'  ==>  Performance is up to %f Gflops, occurred at step %d / %d; time per run = %g sec." % (
        best_config,
        compute_gflops(tuner.task.flop, tuner.task.best.timecost),
        tuner.task.best.occur,
        num_trials,
        tuner.task.best.timecost))

      if hasattr(tuner, 'cleanup'):
        tuner.cleanup()
    else:
      raise Exception('Unrecognized tuner type: `%s`' % tuner_type)
    exit(0)
  else:
    if os.environ['OP'] == 'auto.generic':
      saved_code = codehub_db(os.environ['COMPUTE_V1'])
      if saved_code is not None:
        print("  >> Using Saved Code from Codehub:")
        print("===========================")
        print(saved_code)
        print("===========================")
        exit(0)
    best_config = ''

  assert isinstance(best_config, str)
  if verbose:
    print("====>> [Current Config Option]", best_config)
  if best_config.startswith('['):
    from tvm import auto_scheduler
    origin_cfg = json.loads(best_config)
    origin_cfg = {
      "i": [['["main_compute.<locals>.auto_template"]', 'cuda -keys=cuda,gpu -max_num_threads=%d -thread_warp_size=%d' % (
                device_properties().max_threads_per_block, device_properties().warp_size
             )], origin_cfg],
      "r": [[0], 0, 0, 0],
      "v": "v0.2",
    }
    origin_cfg_file = local_get_dir_file('my_kernel.cfg')
    with open(origin_cfg_file, 'w') as fp:
      fp.write(json.dumps(origin_cfg))
    origin_cfg = tvm.auto_scheduler.measure_record.load_records(origin_cfg_file)
 
    @auto_scheduler.register_workload
    def auto_template():
      _, arg_bufs = default_tune_op.get_template_op()
      return arg_bufs

    target = tvm.target.Target("cuda")
    auto_task = auto_scheduler.create_task(auto_template, (), target)
    for inp, res in origin_cfg:
      s, arg_bufs = auto_task.compute_dag.apply_steps_from_state(inp.state)
      break
  else:
    config = json_to_config(json.loads(best_config)) if best_config else task.config_space
    with ApplyConfig(config):
      with tvm.target.Target(tvm_target):
        s, arg_bufs = default_tune_op.get_template_op()

  device_source, kernel_path = get_target_source(s, arg_bufs)

  if code_only:
    return device_source

  if verbose:
    print("====================================")
    print(device_source)
    print("====================================\n")

  dev_id = int(os.environ.get('DEV_KEY', '0'))
  result = evaluate_perf(kernel_path, task.flop, dev_id)
  exit(0 if result is not None else 1)


if __name__ == '__main__':

  if os.environ.get('HTTP_SERVICE', ''):
    import tornado
    import tornado.httpserver
    import tornado.ioloop
    import tornado.web

    task_lists = collections.deque()

    def clear_environ(compute_exp, step):
      os.environ['COMPUTE_V1'] = compute_exp
      os.environ['OP'] = 'auto.generic'
      os.environ['STEP'] = str(step)
      os.environ['LL_IR'] = ''
      os.environ['COMMIT'] = 'force'

    class IndexHandler(tornado.web.RequestHandler):
      @tornado.gen.coroutine
      def get(self):
        compute_exp = self.request.headers.get('COMPUTE_V1', '')
        num_step = self.request.headers.get('STEP', '')
        print(">> New connection from peer: `%s` (step = %s)" % (compute_exp, num_step))

        if num_step == '@':
          code = '\n'.join(['Steps: %d; Exprs: %s' % (s, c) for s, c in task_lists])
        elif num_step not in ('', '0'):
          task = (int(num_step), compute_exp)
          if task not in task_lists:
            task_lists.append(task)
          code = '[Async Task Has Been Put in Background ..]'
        else:
          code = codehub_db(compute_exp)
          if code is None:
            clear_environ(compute_exp, 0)
            try:
              code = main_compute(code_only=True)
            except:
              print('>> Kernel code failed to generate.')
              code = '[ERROR] ' + traceback.format_exc()
        self.write(code)
        self.flush()
        print(">> Finish subprocess.")
        # yield tornado.gen.sleep(2)

    app = tornado.web.Application([
        (r"/", IndexHandler),
      ],
      cookie_secret = str(random.random()),
      debug = False,
    )
    app.port = int(os.environ.get('HTTP_PORT', '8880'))

    print("* Antares service for backend = `%s` is listening on ':%d'" % (backend, app.port))
    tornado.httpserver.HTTPServer(app).listen(app.port)

    def scan_tasks(ioloop):
      try:
        if os.wait3(os.WNOHANG)[0] != 0:
          task_lists.popleft()
          raise ChildProcessError
        ## Still waiting for current task to complete
      except ChildProcessError:
        if task_lists:
          task_step, task_expr = task_lists[0]
          clear_environ(task_expr, task_step)
          os.environ['HTTP_SERVICE'], _ = '', os.environ['HTTP_SERVICE']
          os.spawnlp(os.P_NOWAIT, '/bin/bash', 'bash', '%s/run.sh' % os.path.dirname(os.path.abspath(__file__)))
          os.environ['HTTP_SERVICE'] = _
      ioloop.add_timeout(time.time() + 5, lambda: scan_tasks(ioloop))

    ioloop = tornado.ioloop.IOLoop.current()
    scan_tasks(ioloop)
    ioloop.start()
  else:
    try:
      main_compute()
    except SystemExit:
      sys.exit(0)
    except:
      traceback.print_exc()
