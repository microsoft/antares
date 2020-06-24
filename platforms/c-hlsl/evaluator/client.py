# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import urllib.request

def eval(kernel_path, **kwargs):
    if not os.environ.get('AGENT_URL', ''):
        print("Skip to evaluator performance: env_var `AGENT_URL` not defined (required: e.g. export AGENT_URL=<ip>:6000)")
        os._exit(1)

    tune_agent_url = 'http://' + os.environ['AGENT_URL']
    with open(kernel_path, 'rb') as fp:
        kernel_data = fp.read()

    try:
      req = urllib.request.Request(tune_agent_url, headers={}, data=kernel_data, method='PUT')
      with urllib.request.urlopen(req) as fp:
          output_content = fp.read().decode()
    except:
      raise Exception("Didn't get correct response from Antares Agent: Bad kernel code, or bad agent address?")

    results = {}
    for line in output_content.split('\n'):
        if line.startswith('- '):
            key, val = line[2:].split('=')
            key = key.strip()
            try:
              results[key] = float(val.strip())
            except:
              results[key] = float('nan')

    # Incorrect result, deny this result
    if 'K/0' not in results or abs(float(results['K/0'])) < 1e-6:
        results = {}
    return results
