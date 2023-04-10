#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time, os, sys, json, socket, subprocess

HOST_ADDR = sys.argv[1].split(':')


def receive(conn, size):
  buff = b''
  while size > 0:
    data = conn.recv(size)
    if not data:
      return None
    size -= len(data)
    buff += data
  return buff

def receive_int(conn):
  val = receive(conn, 32)
  return int(val) if val is not None else None

def send_int(conn, val):
  conn.sendall(('%032u' % val).encode('utf-8'))

def receive_str(conn):
  length = receive_int(conn)
  if length is None:
    return None
  val = receive(conn, length)
  return val if val is not None else None

def send_str(conn, val):
  val = val.encode('utf-8') if isinstance(val, str) else val
  send_int(conn, len(val))
  conn.sendall(val)

def main():
  while True:
    while True:
      try:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((HOST_ADDR[0], int(HOST_ADDR[1])))
        break
      except:
        time.sleep(5)
    print(f"Received connection from peer.")
    while True:
      cmd = receive_str(conn)
      if cmd is None:
        print('Connection closed by peer.')
        break
      print('[*] Receive Command:', cmd.decode('utf-8'))
      if cmd == b'eval':
        print(f'EVAL')
        kernel_data = receive_str(conn)
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
        send_str(conn, json.dumps(results))
      elif cmd == b'mkdirs':
        path = receive_str(conn)
        print(f'MKDIR {path.decode("utf-8")}')
        try:
          os.makedirs(path)
        except:
          pass
      elif cmd == b'upload':
        path = receive_str(conn)
        print(f'UPLOAD {path.decode("utf-8")}')
        data = receive_str(conn)
        with open(path, 'wb') as fp:
          fp.write(data)
      elif cmd == b'download':
        path = receive_str(conn)
        with open(path, 'rb') as fp:
          data = fp.read()
        send_str(conn. data)
      elif cmd == b'msg':
        msg = receive_str(conn)
        print(f'MSG {msg.decode("utf-8")}')
      else:
        print('Unknown cmd:', cmd)


if __name__ == "__main__":
  main()
