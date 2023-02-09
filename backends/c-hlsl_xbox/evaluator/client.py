# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import urllib.request

rev_port = int(os.environ.get('REV', 0))
rev_state = {}

def init(**kwargs):
  if rev_port:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(f"\n>> Waiting for peer to connect to port: {rev_port} ..")
    s.bind(('0.0.0.0', rev_port))
    s.listen()
    conn, addr = s.accept()
    rev_state["s"], rev_state["conn"] = s, conn
    print(f"Received connection from peer.")
  elif not os.environ.get('AGENT_URL', ''):
    print("Skipping evaluation: environment variable `AGENT_URL` not specified (required: e.g. export AGENT_URL=<win10-ip-addr>)")
    exit(1)

def eval(kernel_path, **kwargs):
  if rev_port:
    with open(kernel_path, 'rb') as fp:
      kernel_data = fp.read()
    if int(kwargs.get('compile', 0)):
      import binascii
      return {'HEX': '@' + binascii.hexlify(kernel_data).decode('utf-8') + '@'}
    def receive(s, size):
      buff = b''
      while size > 0:
        data = s.recv(size)
        if not data:
            return b''
        size -= len(data)
        buff += data
      return buff

    conn = rev_state["conn"]
    conn.sendall(('%08u' % len(kernel_data)).encode('utf-8'))
    conn.sendall(kernel_data)
    nbytes = int(receive(conn, 8))
    resp = json.loads(receive(conn, nbytes))
    return resp

  url_with_port = os.environ['AGENT_URL'].strip()
  if ':' not in url_with_port and not url_with_port.endswith('/'):
    url_with_port += ':8860'
  tune_agent_url = 'http://' + url_with_port
  with open(kernel_path, 'rb') as fp:
    kernel_data = fp.read()

  try:
    req = urllib.request.Request(tune_agent_url, headers={
      'ET': str(kwargs['expected_timeout']),
      'OT': os.environ.get('AGENT_OT', '5'),
      'SPECIAL': os.environ.get('SPECIAL', '0'),
      'DEV': str(kwargs['dev_id']),
    }, data=kernel_data, method='PUT')
    with urllib.request.urlopen(req) as fp:
      output_content = fp.read().decode()
  except:
    raise Exception("Didn't get correct response from Antares Agent: Bad kernel code, or bad agent address?")

  start = output_content.find('\n- ')
  if start < 0:
    print(f"Evaluation Error: {output_content}")
    return {}
  stop = output_content.index('\n', start + 1)
  results = output_content[start + 3:stop].strip()
  results = json.loads(results)

  # Incorrect result, deny this result
  if 'K/0' not in results:
    results = {}
  for i in range(len(results)):
    key = 'K/%d' % i
    if key not in results:
      break
    results[key] = float('%.10e' % float(results[key]))
  return results
