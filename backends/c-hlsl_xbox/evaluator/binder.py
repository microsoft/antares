# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time, os, sys, json, socket, subprocess

HOST_ADDR = sys.argv[1].split(':')

def receive(s, size):
    buff = b''
    while size > 0:
        data = s.recv(size)
        if not data:
            return b''
        size -= len(data)
        buff += data
    return buff

while True:
    while True:
      try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST_ADDR[0], int(HOST_ADDR[1])))
        break
      except:
        time.sleep(5)
    print(f"Received connection from peer.")
    while True:
      nbytes = receive(s, 8)
      if not nbytes:
          print('Connection closed by peer.')
          break
      nbytes = int(nbytes)
      kernel_data = receive(s, nbytes)
      try:
          os.remove('my_kernel.cc')
      except:
          pass
      if kernel_data:
          with open('my_kernel.cc', 'wb') as fp:
              fp.write(kernel_data)
          st, output = subprocess.getstatusoutput('evaluator.exe')
      results = {}
      if st != 0:
          print('[Error]', output)
      else:
        for line in output.split('\n'):
          if not line.startswith('- '):
              continue
          k, v = line[2:].split(': ')
          try:
            results[k] = float(v)
          except:
            results[k] = v
      resp = json.dumps(results).encode('utf-8')
      s.sendall(("%08u" % len(resp)).encode('utf-8'))
      s.sendall(resp)
