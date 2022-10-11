# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys, time, math
import subprocess
import hashlib
import traceback

EVAL_PROPERTIES={}

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

    if backend_root:
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

      compile_flags = f'-D__BACKEND__=\\"{backend}\\" -D__BACKEND_{backend[backend.index("-")+1:]}__ -std=c++17 -Wno-string-compare -Wno-unused-result -Wno-unused-value {eval_flags}'
      EVAL_PROPERTIES['compiler'], EVAL_PROPERTIES['compile_flags'] = compiler, compile_flags
      compile_flags += f' -I{backend_root}/include'

      if 0 != os.system(f"diff {backend_root}/include/backend.hpp {os.environ['ANTARES_DRIVER_PATH']}/backend.hpp__{backend} >/dev/null 2>&1"):
        error_info = f"SDK for `{backend}` is not configured correctly, please look into the error messages and reconfigure the corresponding environment."
        compile_cmd = f'{compiler} {source_root}/run_graph.cpp -o {evaluator_path}.tmp {compile_flags}'
        sys.stdout.write('\033[91m')
        print(f'\n[EvalAgent] Compiling Evaluator: {compile_cmd}')
        compile_stat = os.system(f'timeout 30s {compile_cmd}')
        sys.stdout.write('\033[0m\n')
        assert compile_stat == 0, error_info
        os.system(f"cp {backend_root}/include/backend.hpp {os.environ['ANTARES_DRIVER_PATH']}/backend.hpp__{backend}")
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
    with open(evaluator_path, 'rb') as fp:
      exec_magic = fp.read(2)

    if is_wsl == 0 and exec_magic == b'MZ':
      print(f"Antares should run under WSL-1/2 for this backend({backend}), otherwise, evaluation would be skipped.")
      exit(1)

    launcher = f'{backend_root}/launcher.sh'
    if not os.path.exists(launcher):
      launcher = ''
    flags = []
    if int(kwargs.get("compile", 0)) == 0:
      flags += ['--dev', str(dev_id)]
      if int(os.environ.get('PROGRESS', 0)) > 0:
        flags += ['--progress']
      if int(os.environ.get('AB_DEBUG', 0)) > 0:
        flags += ['--debug']
      value_absdir = os.environ.get('VALUE_PATH', '').strip()
      if value_absdir:
        flags += ['--value_absdir', value_absdir]
      timeout = str(kwargs.get("expected_timeout", "")).strip()
      if timeout:
        flags += ['--timeout', timeout]
    else:
      flags += ['--compile']

    if 'VAMAP' in os.environ:
      flags += ['--vamap', os.environ['VAMAP']]

    flags = ' '.join(flags)
    exec_cmd = f'sh -c "cd {os.path.dirname(kernel_path)} && BACKEND={backend} {launcher} {evaluator_path} my_kernel.cc {flags}" || true'
    try:
      output = subprocess.check_output(exec_cmd, shell=True).decode()
    except:
      output = ''

    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            val = val.strip()
            if val[-1].isdigit():
              results[key] = float(val)
            else:
              results[key] = val
    return results
