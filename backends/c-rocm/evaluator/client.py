# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, time, math
import numpy as np
import subprocess

from antares.common import backend

def eval(kernel_path, **kwargs):
    dev_id = kwargs['dev_id']
    source_root = os.path.dirname(__file__)
    source_file = '%s/run_graph.cpp' % source_root

    evaluator_path = '%s/evaluator.%s' % (os.environ['ANTARES_DRIVER_PATH'], backend)
    if not os.path.exists(evaluator_path):
      error_info = f"SDK for `{backend}` is not found, please setup the corresponding environment."
      if backend == 'c-rocm':
        assert 0 == os.system(f'timeout 10s g++ {source_file} -I{source_root} -std=c++17 -Wno-unused-result -lpthread -lamdhip64 -D__HIP_PLATFORM_HCC__ -I/opt/rocm/include -L/opt/rocm/lib -o {evaluator_path}.tmp'), error_info
      elif backend == 'c-cuda':
        assert 0 == os.system(f'timeout 10s g++ {source_file} -I{source_root} -std=c++17 -Wno-unused-result -lpthread -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o {evaluator_path}.tmp'), error_info
      elif backend in ['c-mcpu', 'c-scpu']:
        assert 0 == os.system(f'timeout 10s g++ {source_file} -I{source_root} -std=c++17 -Wno-unused-result -lpthread -ldl -o {evaluator_path}.tmp'), error_info
      else:
        raise Exception("Unrecognized backend type: `%s`" % backend)
      os.system('mv %s.tmp %s >/dev/null 2>&1' % (evaluator_path, evaluator_path))
      assert os.path.exists(evaluator_path)

    exec_cmd = 'sh -c "cd %s && CUDA_VISIBLE_DEVICES=%d EXPECTED_TIMEOUT=%s %s"' % (os.path.dirname(kernel_path), dev_id, kwargs['expected_timeout'], evaluator_path)
    try:
      output = subprocess.check_output(exec_cmd, shell=True).decode()
    except:
      raise Exception("Invalid runtime kernel execution: %s\n" % (exec_cmd))

    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
