# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import importlib
import urllib.request


def eval(kernel_path, **kwargs):
    curr_dir = os.getcwd()
    os.chdir(os.path.dirname(kernel_path))
    assert 0 == os.system('ln -sf %s/run_graph.cpp .' % os.path.dirname(__file__))
    assert 0 == os.system('g++ run_graph.cpp -std=c++14 -lpoplar -lpoplin -lpopnn -lpopops -lpoputil -o run_graph'), "Poplar SDK is not found, please setup the graphcore environment."
    try:
      output = subprocess.check_output('./run_graph', shell=True)
    except:
      raise Exception("Invalid runtime kernel execution: sh -c 'cd %s && ./run_graph'" % os.path.dirname(kernel_path))
    os.chdir(curr_dir)

    output = output.decode()
    results = {}
    for line in output.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split(': ')
            results[key] = float(val)
    return results
