#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os, sys
import pathlib

def main():
  dirname = pathlib.Path(__file__).resolve().parent
  os.environ['PYTHON_EXEC'] = sys.executable
  os.environ['WORKDIR'] = os.path.abspath(os.getcwd())
  os.chdir(dirname)
  cmd = './antares/run.sh'
  os.execl(cmd, cmd, *sys.argv[1:])

if __name__ == '__main__':
  main()
