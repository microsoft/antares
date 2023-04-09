#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import sys, os
import urllib.request
from glob import glob

rev_port = int(os.environ.get('REV', 0))
rev_state = {}

def init():
  import socket
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  print(f"\n>> Waiting for peer to connect to port: {rev_port} ..")
  s.bind(('0.0.0.0', rev_port))
  s.listen()
  conn, addr = s.accept()
  rev_state["s"], rev_state["conn"] = s, conn
  print(f"Received connection from peer.")

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
  val = receive(conn, 8)
  return int(val) if val is not None else None

def send_int(conn, val):
  conn.sendall(('%08u' % val).encode('utf-8'))

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


def eval(kernel_path, **kwargs):
    with open(kernel_path, 'rb') as fp:
      kernel_data = fp.read()
    if int(kwargs.get('compile', 0)):
      import binascii
      return {'HEX': '@' + binascii.hexlify(kernel_data).decode('utf-8') + '@'}

    
    send_str(conn, 'eval')
    send_str(conn, kernel_data)
    resp = receive_str(conn)
    resp = json.loads(resp)
    return resp

if __name__ == "__main__":
  if len(sys.argv) != 3 or sys.argv[1] not in ('upload', 'download'):
    print(f'Usage: REV=.. {sys.argv[0]} [upload|download] <path>')
    exit(0)
  cmd, path = sys.argv[1], sys.argv[2]
  init()
  conn = rev_state["conn"]

  if cmd == 'upload':
    while path.endswith(os.sep):
      path = path[:-len(os.sep)]
    if os.sep not in path:
      path = '.' + os.sep + path
    assert os.path.isdir(path), "Not a directory."
    name = os.path.basename(path)
    assert len(name) > 0, f"Directory with empty name."
    lst = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*'))]
    name = os.path.basename(path)
    if not name:
      name = '.'
    for x in lst:
      if not os.path.isfile(x):
        continue
      assert x.startswith(path)
      with open(x, 'rb') as fp:
        data = fp.read()
      x = x[len(path):]
      if not x.startswith(os.sep):
        x = os.sep + x
      x = name + x
      print(x)
      send_str(conn, 'mkdirs')
      send_str(conn, os.path.dirname(x))
      send_str(conn, 'upload')
      send_str(conn, x)
      send_str(conn, data)
    send_str(conn, 'msg')
    send_str(conn, 'Completed!')
