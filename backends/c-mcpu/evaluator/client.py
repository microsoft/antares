# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import importlib
import urllib.request

try:
    eval_agent = importlib.import_module('backends.c-mcpu.evaluator.eval_agent.eval_agent')
except ModuleNotFoundError:
    raise Exception('>> eval agent is not found')
except:
    traceback.print_exc()
    exit(1)

def eval(kernel_path, **kwargs):
    with open(kernel_path, 'rb') as fp:
        kernel_data = fp.read()

    output_content = ''
    if not os.environ.get('AGENT_URL', ''):
        curr_dir = os.getcwd()
        os.chdir(os.path.join(curr_dir, 'backends/c-mcpu/evaluator/eval_agent'))
        ret, output_content = eval_agent.profile_kernel(kernel_data.decode())
        os.chdir(curr_dir)
    else:
        tune_agent_url = 'http://' + os.environ['AGENT_URL']
        req = urllib.request.Request(tune_agent_url, headers={}, data=kernel_data, method='PUT')
        with urllib.request.urlopen(req) as fp:
            output_content = fp.read().decode()

    results = {}
    for line in output_content.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split('=')
            key = key.strip()
            results[key] = float(val.strip())

    return results
