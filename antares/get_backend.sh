#!/bin/bash -e

if [[ "$BACKEND" == "" ]]; then
  if [ -e /dev/nvidia-modeset ]; then
    BACKEND=c-cuda
  elif [ -e /dev/kfd ]; then
    BACKEND=c-rocm
  elif grep Microsoft /proc/sys/kernel/osrelease >/dev/null; then
    BACKEND=c-hlsl_win64
  fi
fi

echo "${BACKEND:-c-cuda}"
