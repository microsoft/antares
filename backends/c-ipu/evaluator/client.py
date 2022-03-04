# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import importlib
import urllib.request


def init(**kwargs):
    if os.system('popc --version | grep "version 2" >/dev/null') == 0:
        ipu_version = 2
    else:
        ipu_version = 1

    evaluator_path = '%s/evaluator.c-ipu' % os.environ['ANTARES_DRIVER_PATH']
    evaluator_dir = os.path.dirname(evaluator_path)
    try:
        os.makedirs(evaluator_dir)
    except FileExistsError:
        pass
    assert 0 == os.system(f'g++ {os.path.dirname(__file__)}/run_graph.cpp -D__IPU_ARCH_VERSION__={ipu_version} -std=c++14 -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -o {evaluator_path}'), "Poplar SDK is not found, please setup the graphcore environment."

def eval(kernel_path, **kwargs):
    evaluator_path = '%s/evaluator.c-ipu' % os.environ['ANTARES_DRIVER_PATH']

    try:
      cmd = f"sh -c 'pkill -9 evaluator.c-ipu; timeout 60 {evaluator_path} {kernel_path}'"
      output = subprocess.check_output(cmd, shell=True)
    except:
      raise Exception(f"Invalid runtime kernel execution: {cmd}")

    output = output.decode()
    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
