# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys, time, subprocess, os
import random
import hashlib
import traceback
import numpy as np
import math
import re
import json
import importlib
import signal
import collections

import tvm

from antares.common import *
from lang.generic import custom_dtypes, refactor_special_names
from graph_evaluator import client as eval_client

compiler_path = os.path.dirname(os.path.abspath(__file__))

AntaresGlobal.cleanup_funcs = []

def cleanup_on_exit(signum, frame):
  for func in AntaresGlobal.cleanup_funcs:
    try:
      func()
    except:
      pass
  exit(0 if signum == -1 else 1)

signal.signal(signal.SIGINT, cleanup_on_exit)

tvm_target = 'cuda'
verbose = int(os.environ.get('VERBOSE', '1'))

try:
  backend_config = importlib.import_module('backends.%s.config' % backend)
  backend_root = os.path.dirname(backend_config.__file__)
except ModuleNotFoundError:
  raise Exception('>> Platform config for backend %s not found' % backend)
except:
  traceback.print_exc()
  exit(1)

def get_global_arg_props():
  return AntaresGlobal.global_arg_pros

def device_properties():
  return AntaresGlobal.attrs.device_props

def verify_body(kernel_name, body):
  max_threads_per_block = device_properties().max_threads_per_block
  max_shared_memory_per_block = device_properties().max_shared_memory_per_block
  assert max_threads_per_block > 0 and max_shared_memory_per_block >= 0, '[Error] Invalid device properties, maybe device is not detected correctly.'

  thread_extents, shared_mem_in_bytes = dict(), 0
  thread_extent_symbol, shared_symbol = '// [thread_extent] ', '__shared__ '
  for line in body.split('\n'):
    ll = line.strip()
    if ll.startswith(thread_extent_symbol):
      key, val = ll[len(thread_extent_symbol):].split(' = ')
      if key not in thread_extents:
        thread_extents[key] = int(val)
      else:
        assert thread_extents[key] == int(val), "Inequivalent thread_extents in function `%s`: %d v.s. %d" % (kernel_name, thread_extents[key], int(val))
    if ll.startswith(shared_symbol):
      assert ll.endswith('];');
      ctype, _, count = ll[len(shared_symbol):-2].replace('[', ' ').split()
      if ctype in ('double', 'long', 'int64_t'):
        shared_mem_in_bytes += int(count) * 8
      elif ctype in ('float', 'int'):
        shared_mem_in_bytes += int(count) * 4
      elif ctype in ('half', 'short'):
        shared_mem_in_bytes += int(count) * 2
      elif ctype in ('bool', 'char'):
        shared_mem_in_bytes += int(count) * 1
      else:
        raise Exception("Unrecoginized C datatype: %s" % ctype)

  num_threads = thread_extents.get('threadIdx.x', 1) * thread_extents.get('threadIdx.y', 1) * thread_extents.get('threadIdx.z', 1)
  assert num_threads <= max_threads_per_block, "Invalid num_threads used in function `%s`: num_threads(%d) > max_threads_per_block(%d)" % (kernel_name, num_threads, max_threads_per_block)
  assert shared_mem_in_bytes <= max_shared_memory_per_block, "Invalid shared memory used in function `%s`: used_shared_mem_in_bytes %d > max_shared_memory_per_block %d" % (kernel_name, shared_mem_in_bytes, max_shared_memory_per_block)

def translate_code(code, config):
  global_arg_props = get_global_arg_props()

  code = refactor_special_names(code, global_arg_props)
  tensors_pool = json.loads(os.environ.get('TENSORS_POOL', '{}'))
  kernel_slices = []
  for kernel in ('\n' + code).split('\nextern ')[1:]:
    kernel = 'extern %s\n' % kernel[:kernel.index('\n}') + 2]
    idx = kernel.index(' void ') + 6
    idy = kernel.index('(', idx)
    kernel_name = kernel[idx:idy]
    kernel_prefix = 'template_op_kernel'
    assert kernel_name.startswith(kernel_prefix)
    kernel_id = int(kernel_name[len(kernel_prefix):])
    idx = kernel.index(') {', idy)
    body = kernel[idx+3:kernel.index('\n}', idx)].strip()
    verify_body(kernel_name, body)

    arg_line = kernel[idy+1:idx]
    args, outputs_ex = [], []
    for x in arg_line.split(','):
      c_type = x.split('*')[0].strip()
      v_name = x.split()[-1]
      if v_name.startswith('___'):
        continue
      v_name_in_pool = v_name[2:] if re.match(r'^__[a-z]+', v_name) else v_name
      v_props = tensors_pool[v_name_in_pool]
      if re.search(r'\b___%s\b' % v_name, arg_line) is not None:
        outputs_ex.append((c_type, v_name, v_props))
      else:
        args.append((c_type, v_name, v_props))
    kernel_slices.append((kernel_id, kernel_name, args + outputs_ex, body))
  return kernel_slices


def compute_gflops(flop, t):
  try:
    return flop / (t * 1e3) / 1e6
  except:
    return 0.0

def codehub_db(compute_key, source_code=None, erase=False):
  compute_key = compute_key.split('##')[0].strip()
  digest = hashlib.sha256(compute_key.encode()).hexdigest()
  os.system('mkdir -p %s/../codehub' % compiler_path)
  code_path = '%s/../codehub/%s.%s' % (compiler_path, digest, backend)
  if erase:
    try:
      os.remove(code_path)
    except:
      pass
    return None
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

def get_target_source(best_config, dir_sid=None):
  # Note: Not thread safe due to multiple invokes of target codegen

  global_arg_props = get_global_arg_props()
  def get_kernel_metadata(config):
    inp_args, outp_args = [], []

    for buf in global_arg_props['_in']:
      if buf['name'].startswith('_'):
        # Just for Auto Shard
        assert(buf['dtype'] == 'int32' and buf['shape'] == [1])
        continue
      inp_args.append('%s:%s%s' % (buf['name'], buf['dtype'], buf['shape']))
    for buf in global_arg_props['_out']:
      outp_args.append('%s:%s%s' % (buf['name'], buf['dtype'], buf['shape']))

    device_code = os.environ.get('DEVICE_NAME', '')
    device_code = device_code if device_code else 'default'
    header_meta = '// GLOBALS: ' + ', '.join(inp_args) + ' -> ' + ', '.join(outp_args) + '\n// BACKEND: %s (%s)\n' % (backend, device_code)
    properties = "// CONFIG: %s\n// COMPUTE_V1: %s\n" % (config.strip() if isinstance(config, str) else '', os.environ['COMPUTE_V1'])
    return header_meta + properties

  def slices_to_code(kernel_slices):
    def tensor_display(encoded_name, prop):
      return f'{encoded_name}:{prop["dtype"]}{str(prop["shape"])}'

    kernel_slices.sort()
    code = ['']
    for i, (kernel_id, kernel_name, args, body) in enumerate(kernel_slices):
      num_outputs = len(global_arg_props['_out']) if i + 1 == len(kernel_slices) else 1
      display_inputs = ', '.join([tensor_display(x, prop) for _, x, prop in args[:-num_outputs]])
      display_outputs = ', '.join([tensor_display(x, prop) for _, x, prop in args[-num_outputs:]])
      kernel = backend_config.do_native_translation_v2((kernel_name, args[:-num_outputs], args[-num_outputs:], body), attrs=AntaresGlobal.attrs).strip()
      code.append(f'// LOCAL: {kernel_name} -- {display_inputs} -> {display_outputs}\n\n{kernel}\n')

    del kernel_slices
    code = '\n// ---------------------------------------------------------------------------\n'.join(code)
    return code

  def pack_device_source(kernel_slices):
    device_source = slices_to_code(kernel_slices)
    device_source = '%s\n%s' % (get_kernel_metadata(best_config), device_source)
    kernel_path = local_get_dir_file('my_kernel.cc', dir_sid=dir_sid)
    with open(kernel_path, 'w') as fp:
      fp.write(device_source)
    return device_source, kernel_path

  if getattr(AntaresGlobal, 'mode', None) == 'antares':
    json_config = json.loads(best_config)
    kernel_slices = backend_config.to_kernel_slices(AntaresGlobal.compute_graph, json_config if json_config is not None else {})
    return pack_device_source(kernel_slices)

  with open(local_get_dir_file('my_kernel.time', dir_sid=dir_sid), 'w') as fp:
    fp.write('%s' % time.time())
  default_tune_op = AntaresGlobal.default_tune_op
  assert isinstance(best_config, str), "Config value must be string type, got: %s" % best_config.__class__
  if best_config.startswith('['):
    # Ansor config
    from tvm import auto_scheduler
    [origin_cfg] = json.loads(best_config)
    origin_cfg_file = local_get_dir_file('my_kernel.cfg', dir_sid=dir_sid)
    with open(origin_cfg_file, 'w') as fp:
      fp.write(json.dumps(origin_cfg))
    origin_cfg = tvm.auto_scheduler.measure_record.load_records(origin_cfg_file)

    from tuner.Ansor.main import create_auto_task
    target = tvm.target.Target(tvm_target)
    auto_task = create_auto_task(target)

    for inp, res in origin_cfg:
      s, arg_bufs = auto_task.compute_dag.apply_steps_from_state(inp.state)
      break
    with open(local_get_dir_file('my_kernel.sched', dir_sid=dir_sid), 'w') as fp:
      fp.write(auto_task.compute_dag.print_python_code_from_state(inp.state))
  else:
    AntaresGlobal.attrs.auto_config.set_candidate(json.loads(best_config))
    with tvm.target.Target(tvm_target):
      s, arg_bufs = default_tune_op.get_template_op()

  if s is not None:
      lower_source = str(tvm.lower(s, arg_bufs, simple_mode=True))

      lower_file = local_get_dir_file('my_kernel.lower', dir_sid=dir_sid)
      with open(lower_file, 'w') as fp:
        fp.write(lower_source)

      # Compile Source Code
      def build_template():
        return tvm.build(s, arg_bufs, tvm_target, name='template_op')
      func = build_template()

  assert(len(func.imported_modules) == 1)
  kernel_slices = translate_code(func.imported_modules[0].get_source(), best_config)
  return pack_device_source(kernel_slices)

def code_suffix(tpr=-1.0, step_prod=0, step_plan=-1):
  return '\n// Saved Perf = %.6e sec / run; Step Produced = %d; Planned Steps = %d;' % (tpr, step_prod, step_plan)

def evaluate_perf(kernel_path, dev_id, device_source, dir_sid=None, verbose=True, expected_timeout=None):
  if verbose:
    print("\n[EvalAgent] Evaluating Modules..\n")

  def handle_result(result):
    if verbose:
      print('\n[EvalAgent] Results =', json.dumps(result))
    if 'RESULT' in os.environ:
      if abs(float(os.environ['RESULT']) / result['K/0'] - 1.0) > 1e-6:
        result['TPR'] = None

    t = result.get('TPR', None)
    if 'K/0' in result and t is None:
      t = result['TPR'] = float('inf')
    if t is None:
      print("\n[Antares] Incorrect compute kernel from evaluator.")
    else:
      gflops = compute_gflops(AntaresGlobal.default_task.flop, t)
      if verbose:
        print("\n[Antares] Average time cost / run = %g sec, %g gflops." % (t, gflops))
      with open(local_get_dir_file('result.txt', dir_sid=dir_sid), 'w') as fp:
        fp.write(str(t) + '\n')
        for i in range(len(result)):
          key = 'K/%d' % i
          if key not in result:
            break
          fp.write(str(result[key]) + '\n')
    if os.environ.get('COMMIT', ''):
      kernel_path = codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + code_suffix(tpr=t))
      print('  >> Update current code to codehub: %s' % kernel_path)

  def do_evaluate(expected_timeout):
    try:
      if expected_timeout is None:
        expected_timeout = os.environ.get('EXPECTED_TIMEOUT', 'inf')
      if expected_timeout in ('', 'inf', float('inf')):
        expected_timeout = ''
      else:
        expected_timeout = float(expected_timeout)
        expected_timeout = max(expected_timeout * 1.1, expected_timeout + 0.1)

      results = eval_client.eval(kernel_path=local_get_dir_file('my_kernel.cc', dir_sid=dir_sid),
                  expected_timeout=expected_timeout,
                  dev_id=dev_id, backend_root=backend_root
                )
      return results
    except SystemExit:
      return None
    except:
      if verbose:
        traceback.print_exc()
      return None

  try:
    results = do_evaluate(expected_timeout)
    if results is not None:
      handle_result(results)
  except:
    pass
  return results

def compute_mem_ratio(tpr):
  if math.isinf(tpr) or math.isinf(float(device_properties().mem_bandwith)) or device_properties().mem_bandwith <= 0:
    return -1

  global_arg_props = get_global_arg_props()
  access_bytes = 0
  for buf in global_arg_props['_in']:
    access_bytes += np.product(buf['shape']) * get_type_size(buf['dtype'])
  for buf in global_arg_props['_out']:
    access_bytes += np.product(buf['shape']) * get_type_size(buf['dtype'])

  access_bytes = int(access_bytes)
  if access_bytes <= 0:
    return -1
  ratio = np.ceil(access_bytes * 1e-7 / tpr / device_properties().mem_bandwith)
  return min(int(ratio), 100)

def run_config_entity(target_source, config_str, dir_sid, expected_timecost='inf', dev_id=0):
  config_str_short = config_str # if len(config_str) < 60 else config_str[:60] + '..'
  print("  >> [ ] Param_entity on sid = %s: config = '%s', dev_id = %d, upper_bound_tpr = %.6e s" % (dir_sid, config_str_short, dev_id, expected_timecost))
  try:
    assert target_source is not None, "Invalid target source detected in verification stage."
    device_source, kernel_path = target_source

    results = evaluate_perf(kernel_path, dev_id, device_source, dir_sid, verbose=False, expected_timeout=expected_timecost)
    assert results is not None and 'TPR' in results, "Invalid target output detected in evaluation stage."
    digest = ','.join(['%.6e' % float(results['K/%d' % i]) for i in range(len(results) - 1)])
    digest = f'\033[{91 + int(hashlib.sha256(digest.encode()).hexdigest(), 16) % 6}m{digest}\033[0m'
    result = float(results['TPR'])
  except:
    digest = 'null'
    result = float('inf')
  if not math.isinf(result):
    print("  >> [*] Param_entity on sid = %s: config = '%s', tpr = `%.6f`, digest = `%s`, mem_occupy = %d %%" % (dir_sid, config_str_short, result, digest, compute_mem_ratio(result)))
  return result


def main_compute(code_only=False):
  def compile_callback(code):
   return bytearray()
  tvm.register_func('tvm_callback_cuda_compile', compile_callback, override=True)

  default_tune_op = importlib.import_module('lang.generic')

  import logging
  from tvm import autotvm

  logging.getLogger('autotvm').setLevel(logging.ERROR)
  logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
  task = autotvm.task.create("template_op", args=(), target=tvm_target)

  AntaresGlobal.default_tune_op = default_tune_op
  AntaresGlobal.default_task = task

  if verbose:
    print('  >> Backend = %s, Python PID = %s, Task = %s;' % (backend, os.getpid(), default_tune_op.__name__))

  num_trials = int(os.environ['STEP']) if 'STEP' in os.environ else 0

  config = os.environ.get('CONFIG', '').strip()
  if config != '':
    best_config = config
  elif num_trials > 0:
    dev_num = backend_config.get_execution_parallism()
    if dev_num <= 0:
        raise Exception("No valid device found for backend: %s." % backend)
    batch_size = os.environ.get('BATCH', '')
    batch_size = 16 if not batch_size else int(batch_size)

    from concurrent.futures import ThreadPoolExecutor
    worker_size = batch_size if batch_size < dev_num else dev_num
    thread_pool = ThreadPoolExecutor(max_workers=worker_size)

    tuner_type = os.environ.get('TUNER')
    if not tuner_type:
      explicit_ops = AntaresGlobal.attrs.explicit_ops
      tuner_type = 'OpEvo'
    print('  >> MAKE_PARA = %d/%d, EXEC_PARA = %d, TUNER = %s\n' % (worker_size, batch_size, dev_num, tuner_type))

    auto_commit = os.environ.get('COMMIT', '')
    if auto_commit:
      saved_code = codehub_db(os.environ['COMPUTE_V1'])
      if saved_code is not None and auto_commit != 'force':
        raise Exception("Saved code has existed in codehub. Please try COMMIT=force to override it.")
      os.environ.pop('COMMIT')

    try:
      if getattr(AntaresGlobal, 'mode', None) == 'antares':
        task.search_space_v2 = backend_config.search_space(AntaresGlobal.compute_graph)
      else:
        task.search_space_v2 = AntaresGlobal.attrs.auto_config.get_config_space()
      task.n_parallel = batch_size
      tuner = importlib.import_module('tuner.%s.main' % tuner_type)
      tuner = tuner.MainTuner(task)
    except:
      raise Exception('>> Cannot import Antares Tuner: %s' % tuner_type)

    if hasattr(tuner, 'cleanup'):
      AntaresGlobal.cleanup_funcs.append(tuner.cleanup)

    if tuner is not None:
      AntaresGlobal.current_step = 0
      eval_client.init(backend_root=backend_root)

      def measure_batch(inputs):
        results, futures = [], []
        target_sources, config_strs = [], []
        for i in range(len(inputs)):
          dir_sid = AntaresGlobal.current_step + i + 1
          config_str = inputs[i].config if type(inputs[i].config).__name__ == 'str' else 'null'
          config_strs.append(config_str)
          try:
            target_source = get_target_source(config_strs[i], dir_sid)
          except:
            # traceback.print_exc()
            target_source = None
          target_sources.append(target_source)

        expected_timecost = tuner.task.best.timecost if not math.isinf(tuner.task.best.timecost) else min(30, float(os.environ.get('EXPECTED_TIMEOUT', 'inf')))
        for i in range(len(inputs)):
          dir_sid = AntaresGlobal.current_step + i + 1
          futures.append(thread_pool.submit(run_config_entity, target_sources[i], config_strs[i], dir_sid, expected_timecost, i % dev_num))

        best_slot = -1
        for i in range(len(inputs)):
          dir_sid = AntaresGlobal.current_step + i + 1
          t = futures[i].result()
          if t < tuner.task.best.timecost:
            best_slot = dir_sid
            tuner.task.best.timecost = t
            tuner.task.best.config = inputs[i].config
            tuner.task.best.occur = best_slot
          results.append(autotvm.measure.MeasureResult(costs=(t,), error_no=0, all_cost=i, timestamp=time.time()))
        AntaresGlobal.current_step += len(results)

        stage_logs = 'STEP[%d / %d] Current Best Config = %s, Perf = %g sec / op (%g Gflops), MemRatio = %g %%, Occur Step = %d;' % (
          AntaresGlobal.current_step, num_trials,
          tuner.task.best.config,
          tuner.task.best.timecost, compute_gflops(tuner.task.flop, tuner.task.best.timecost),
          compute_mem_ratio(tuner.task.best.timecost),
          tuner.task.best.occur)

        print('\n\033[93m%s\033[0m' % ('=' * min(120, len(stage_logs))))
        print(stage_logs)
        print('\033[93m%s\033[0m\n' % ('=' * min(120, len(stage_logs))))

        if auto_commit and best_slot >= 0:
          with open(local_get_dir_file('my_kernel.cc', best_slot), 'r') as fp:
            device_source = fp.read()
          with open(local_get_dir_file('result.txt', best_slot), 'r') as fp:
            t = float(fp.read().split()[0])
          kernel_path = codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + code_suffix(tpr=t, step_prod=best_slot, step_plan=num_trials))
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

      tuner.tune(n_trial=num_trials, callbacks=callbacks, measure_option=None)

      if math.isinf(tuner.task.best.timecost):
        print(f'[Error] No valid config found in the whole tuning. (Try other tuner types other than `TUNER={tuner_type}`?)')
        cleanup_on_exit(0, None)

      best_config = tuner.task.best.config

      if auto_commit:
          device_source = codehub_db(os.environ['COMPUTE_V1'])
          codehub_db(os.environ['COMPUTE_V1'], source_code=device_source + '\n// Antares Tuning Completed in %d steps.' % AntaresGlobal.current_step)

      print("\n[Best Config] CONFIG='%s'  ==>  Performance is up to %f Gflops, occurred at step %d / %d; time per run = %g sec." % (
        best_config,
        compute_gflops(tuner.task.flop, tuner.task.best.timecost),
        tuner.task.best.occur,
        num_trials,
        tuner.task.best.timecost))

      cleanup_on_exit(-1, None)
    else:
      raise Exception('Unrecognized tuner type: `%s`' % tuner_type)
    exit(0)
  else:
    saved_code = codehub_db(os.environ['COMPUTE_V1'])
    if saved_code is not None:
      print("  >> Using Saved Code from Codehub:")
      print("// ---------------------------------------------------------------------------")
      print(saved_code)
      print("// ---------------------------------------------------------------------------")
      exit(0)
    best_config = ''

  assert isinstance(best_config, str)

  best_config = best_config if best_config else 'null'
  device_source, kernel_path = get_target_source(best_config)

  if code_only:
    return device_source

  if verbose:
    print()
    print("// ---------------------------------------------------------------------------")
    print(device_source)
    print("// ---------------------------------------------------------------------------")

  eval_client.init(backend_root=backend_root)
  dev_id = int(os.environ.get('DEV_ID', '0'))
  result = evaluate_perf(kernel_path, dev_id, device_source)
  exit(0 if result is not None and len(result) > 1 else 1)


def rest_service():
  import tornado
  import tornado.httpserver
  import tornado.ioloop
  import tornado.web

  task_lists = collections.deque()

  def clear_environ(compute_exp, step):
      os.environ['COMPUTE_V1'] = compute_exp
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
          codehub_db(compute_exp, erase=True)
          code = '[Async Task Has Been Put in Background ..]'
        else:
          code = codehub_db(compute_exp)
          duplicate_items = [(s, c) for s, c in task_lists if c == compute_exp]

          if code is None:
            clear_environ(compute_exp, 0)
            try:
              code = main_compute(code_only=True)
              if duplicate_items:
                code += code_suffix(tpr=-1.0, step_prod=0, step_plan=duplicate_items[0][0])
            except:
              print('>> Kernel code failed to generate.')
              code = '[ERROR] ' + traceback.format_exc()
          elif '// Antares Tuning Completed in' not in code and not duplicate_items:
              code = code[:code.rindex('\n// Saved Perf')]
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
  try:
    tornado.httpserver.HTTPServer(app).listen(app.port)
  except Exception as ex:
    raise Exception("Error: %s.\n\nService Failure: Is another process listening on TCP-port %d? Try setting another port with: export HTTP_PORT=<port-num>" % (ex, app.port))

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
          os.spawnlp(os.P_NOWAIT, '/bin/bash', 'bash', '%s/run.sh' % compiler_path)
          os.environ['HTTP_SERVICE'] = _
      ioloop.add_timeout(time.time() + 5, lambda: scan_tasks(ioloop))

  ioloop = tornado.ioloop.IOLoop.current()
  scan_tasks(ioloop)
  ioloop.start()


if __name__ == '__main__':
  try:
    if os.environ.get('HTTP_SERVICE', ''):
      rest_service()
    else:
      main_compute()
  except SystemExit:
    cleanup_on_exit(0, None)
  except:
    traceback.print_exc()
