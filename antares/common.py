# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess


class Mock(object):
  pass


def wait_for(func, exit_timeout=None, args=[]):
  if not exit_timeout:
    return func(*args)
  def timeout_handler():
    print("Error: Timeout during Kernel warmup")
    os._exit(1)
  from threading import Timer
  my_timer = Timer(exit_timeout, timeout_handler, [])
  my_timer.start()
  res = func(*args)
  my_timer.cancel()
  del my_timer
  return res

def local_get_dir_file(rel_file, dir_sid=None):
  if dir_sid is None:
    dir_sid = os.environ['DIR_SID'] if 'DIR_SID' in os.environ else '_'
  dir_space = os.path.join(os.environ['ANTARES_DRIVER_PATH'], 'cache')
  os.system('mkdir -p "%s/%s"' % (dir_space, dir_sid))
  return "%s/%s/%s" % (dir_space, dir_sid, rel_file)

def run_process_with_timeout(args, timeout=None, envs=None):
  try:
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=envs)
    retcode = proc.wait(timeout=timeout)
    return retcode == 0
  except subprocess.TimeoutExpired:
    print('Timed out - killing', proc.pid)
    proc.kill()
    return False

def system_lock(key_ids):
  import socket, time
  occupied_sock = None
  while not occupied_sock:
    for key_id in key_ids:
      try:
        sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 9050 + key_id))
        sock.listen(1)
        occupied_sock = (sock, key_id)
        break
      except:
        try:
          sock.shutdown(socket.SHUT_RDWR)
          sock.close()
        except:
          sock.close()
    if occupied_sock:
      break
    # print('still waiting ..')
    time.sleep(0.2)

  # print('Using key_id = %d' % occupied_sock[1])
  sock = occupied_sock[0]

  def unlock_fd():
    try:
      sock.shutdown(socket.SHUT_RDWR)
      sock.close()
    except:
      sock.close()
  return unlock_fd, occupied_sock[1]

def type_to_c(dtype):
  idx = dtype.find('@')
  if idx >= 0:
    return dtype[:idx]
  native_types = {'float32': 'float', 'int32': 'int', 'int8': 'char'}
  if dtype in native_types:
    return native_types[dtype]
  raise Exception("Unhandled ctype mapping case: %s" % dtype)

backend = os.environ['BACKEND'] if 'BACKEND' in os.environ else 'c-rocm'
AntaresGlobal = Mock()

