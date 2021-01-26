# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, time, math
import numpy as np
import subprocess

from antares.common import backend

def eval(kernel_path, **kwargs):
    source_file = '%s/run_graph.cpp' % os.path.dirname(__file__)
    dest_dir = os.path.dirname(kernel_path)
    assert 0 == os.system(f'timeout 10s dpcpp {source_file} -I{dest_dir} -std=c++17 -lpthread -o {dest_dir}/run_graph'), "Compiling error or DPC++ SDK not found, please check the oneAPI environment."
    
    exec_cmd = 'sh -c "cd %s && EXPECTED_TIMEOUT=%s %s"' % (dest_dir, kwargs['expected_timeout'], './run_graph')
    try:
      output = subprocess.check_output(exec_cmd, shell=True).decode()
    except:
      raise Exception("Invalid runtime kernel execution: %s\n" % (exec_cmd))

    print(output)

    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
