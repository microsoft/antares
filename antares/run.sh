#!/bin/bash -e

cd $(dirname $0)/..

if [[ "$(pwd)" != "/antares" ]]; then
  echo "Please run task in Docker environment."
  exit 1
fi

# Valid Backends: c-cuda, c-rocm, c-mcpu, c-hlsl, c-gc

export BACKEND=${BACKEND:-c-rocm}
export ANTARES_DRIVER_PATH=/tmp/libAntares

mkdir -p ${ANTARES_DRIVER_PATH}

if ! diff engine/antares_driver.cc ${ANTARES_DRIVER_PATH}/.antares_driver.cc >/dev/null 2>&1; then
  cp engine/antares_driver.cc ${ANTARES_DRIVER_PATH}/antares_driver.cc
  g++ ${ANTARES_DRIVER_PATH}/antares_driver.cc -std=c++11 -lpthread -ldl -I/opt/rocm/include -D__HIP_PLATFORM_HCC__=1 -O2 -fPIC -shared -o ${ANTARES_DRIVER_PATH}/libcuda.so.1
  ln -sf libcuda.so.1 ${ANTARES_DRIVER_PATH}/libcudart.so.$(echo /usr/local/cuda-* | awk -F\- '{print $NF}')
  mv ${ANTARES_DRIVER_PATH}/antares_driver.cc ${ANTARES_DRIVER_PATH}/.antares_driver.cc
fi

if [[ "${BACKEND}" == "c-cuda" ]]; then
  if ! ls /dev/nvidia[0-9] >/dev/null 2>&1; then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat
    export LD_PRELOAD=${ANTARES_DRIVER_PATH}/libcuda.so.1
  fi
else
  export LD_LIBRARY_PATH=${ANTARES_DRIVER_PATH}
fi

export HIP_PLATFORM=hcc
export HSA_USERPTR_FOR_PAGED_MEM=0

ldconfig >/dev/null 2>&1 || true

[[ "$USING_GDB" == "" ]] || USING_GDB="gdb --ex run --args"

time OP=${OP:-auto.generic} STEP=${STEP:-0} ${USING_GDB} python3 ./antares/antares_compiler.py "$@"
