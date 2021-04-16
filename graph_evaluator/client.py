# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, time, math
import numpy as np
import subprocess
import hashlib
import traceback

def init(**kwargs):
    backend_root = kwargs['backend_root']
    backend = os.path.basename(backend_root)
    source_root = f'{backend_root}/../../graph_evaluator'

    if not os.path.exists(f'{backend_root}/include/backend.hpp'):
      global eval_client
      try:
        import importlib
        eval_client = importlib.import_module('backends.%s.evaluator.client' % backend)
      except ModuleNotFoundError:
        print('\n[EvalAgent] Evaluator for backend `%s` not found, skipping evaluation.' % backend)
        exit(1)
      except:
        traceback.print_exc()
        exit(1)
      return eval_client.init(**kwargs)

    evaluator_path = '%s/evaluator.%s' % (os.environ['ANTARES_DRIVER_PATH'], backend)

    if 0 != os.system(f"diff {backend_root}/include/backend.hpp {os.environ['ANTARES_DRIVER_PATH']}/backend.hpp-{backend} >/dev/null 2>&1"):
      with open(f'{backend_root}/include/backend.hpp', 'r') as fp:
        eval_flags_pref = f'//; eval_flags({backend}):'
        eval_flags, compiler = '', 'g++'
        while True:
          line = fp.readline()
          if not line:
            break
          line = line.strip()
          if line.startswith(eval_flags_pref):
            eval_flags = line[len(eval_flags_pref):].strip()
            if eval_flags.startswith('['):
              idx = eval_flags.index(']')
              eval_flags, compiler = eval_flags[idx+1:].strip(), eval_flags[1:idx].strip()
            else:
              eval_flags += ' -lpthread'
            break

      error_info = f"SDK for `{backend}` is not configured correctly, please look into the error messages and reconfigure the corresponding environment."
      compile_cmd = f'{compiler} {source_root}/run_graph.cpp -D__BACKEND__=\\"{backend}\\" -I{backend_root}/include -std=c++17 -Wno-string-compare -Wno-unused-result -Wno-unused-value -o {evaluator_path}.tmp {eval_flags}'
      print(f'\n[EvalAgent] Compiling Evaluator: {compile_cmd}')
      assert 0 == os.system(f'timeout 30s {compile_cmd}'), error_info
      os.system(f"cp {backend_root}/include/backend.hpp {os.environ['ANTARES_DRIVER_PATH']}/backend.hpp-{backend}")
      os.system(f'mv {evaluator_path}.tmp {evaluator_path} >/dev/null 2>&1')
      is_wsl = 1 if (os.environ.get('IS_WSL', '0') == '1') else 0

def eval(kernel_path, **kwargs):
    dev_id = kwargs['dev_id']
    backend_root = kwargs['backend_root']
    backend = os.path.basename(backend_root)

    evaluator_path = '%s/evaluator.%s' % (os.environ['ANTARES_DRIVER_PATH'], backend)
    if not os.path.exists(evaluator_path):
      global eval_client
      return eval_client.eval(kernel_path, **kwargs)

    is_wsl = 1 if (os.environ.get('IS_WSL', '0') == '1') else 0
    if is_wsl == os.system(f'file {evaluator_path} | grep "MS Windows" >/dev/null 2>&1'):
      print(f"Antares should run under WSL1.0 for this backend({backend}), otherwise, evaluation would be skipped.")
      exit(1)

    launcher = f'{backend_root}/launcher.sh'
    if not os.path.exists(launcher):
      launcher = ''
    exec_cmd = 'sh -c "cd %s && DEV_ID=%d EXPECTED_TIMEOUT=%s %s %s my_kernel.cc" || true' % (os.path.dirname(kernel_path), dev_id, kwargs['expected_timeout'], launcher, evaluator_path)
    try:
      output = subprocess.check_output(exec_cmd, shell=True).decode()
    except:
      output = ''

    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
