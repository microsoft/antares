#!/bin/bash -e

if [[ "$BACKEND" == "" ]]; then
  if [ -e $(dirname $0)/../backend.default ]; then
    BACKEND=$(cat $(dirname $0)/../backend.default)
  elif [ -e /dev/nvidiactl ]; then
    BACKEND=c-cuda
  elif [ -e /dev/kfd ]; then
    BACKEND=c-rocm
  elif grep Microsoft /proc/sys/kernel/osrelease >/dev/null || grep WSL2 /proc/sys/kernel/osrelease >/dev/null; then
    BACKEND=c-hlsl_win64
  fi
fi

echo "${BACKEND:-c-mcpu}"
