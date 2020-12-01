# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, time, math
import numpy as np
import subprocess

from antares.common import backend

def eval(kernel_path, **kwargs):
    dev_id = kwargs['dev_id']
    curr_dir = os.getcwd()
    os.chdir(os.path.dirname(kernel_path))
    source_file = '%s/run_graph.cpp' % os.path.dirname(__file__)

    evaluator_path = '%s/evaluator.%s' % (os.environ['ANTARES_DRIVER_PATH'], backend)
    if not os.path.exists(evaluator_path):
      if backend == 'c-rocm':
        assert 0 == os.system('timeout 10s /opt/rocm/bin/hipcc %s -std=c++17 -lpthread -o %s.tmp' % (source_file, evaluator_path)), "ROCm SDK is not found, please setup the graphcore environment."
      elif backend == 'c-cuda':
        assert 0 == os.system('timeout 10s g++ %s -std=c++17 -lcuda -lcudart -lpthread -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o %s.tmp' % (source_file, evaluator_path)), "CUDA SDK is not found, please setup the graphcore environment."
      else:
        raise Exception("Unrecognized backend type for `%s`" % backend)
      os.system('mv %s.tmp %s >/dev/null 2>&1' % (evaluator_path, evaluator_path))
      assert os.path.exists(evaluator_path)

    exec_cmd = "sh -c 'cd %s && CUDA_VISIBLE_DEVICES=%d EXPECTED_TIMEOUT=%s %s'" % (os.path.dirname(kernel_path), dev_id, kwargs['expected_timeout'], evaluator_path)
    st, output = subprocess.getstatusoutput(exec_cmd)
    os.chdir(curr_dir)
    if st != 0:
        raise Exception("Invalid runtime kernel execution: %s\n\nReason: %s" % (exec_cmd, output))

    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
