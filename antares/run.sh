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

if [[ "$BACKEND" == "c-rocm" ]]; then
  /opt/rocm/bin/hipcc engine/cuda_properties.cc -o ${ANTARES_DRIVER_PATH}/device_properties
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
elif [[ "$BACKEND" == "c-cuda" ]]; then
  /usr/local/cuda/bin/nvcc engine/cuda_properties.cc -o ${ANTARES_DRIVER_PATH}/device_properties
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
else
  rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
fi

if [ ! -e ${ANTARES_DRIVER_PATH}/device_properties.cfg ]; then
  if [[ "${HARDWARE_CONFIG}" != "" ]]; then
    cat hardware/${HARDWARE_CONFIG}.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    echo "  >> Using specific hardware device properties."
  else
    cat platforms/${BACKEND}/default_props.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    echo "  >> Using ${BACKEND} default device properties."
  fi
else
  echo "  >> Using ${BACKEND} runtime device properties."
fi

if ! diff engine/antares_driver.cc ${ANTARES_DRIVER_PATH}/.antares_driver.cc >/dev/null 2>&1; then
  cp engine/antares_driver.cc ${ANTARES_DRIVER_PATH}/antares_driver.cc
  g++ ${ANTARES_DRIVER_PATH}/antares_driver.cc -std=c++11 -lpthread -ldl -I/opt/rocm/include -D__HIP_PLATFORM_HCC__=1 -O2 -fPIC -shared -o ${ANTARES_DRIVER_PATH}/libcuda.so.1
  ln -sf libcuda.so.1 ${ANTARES_DRIVER_PATH}/$(ldd /opt/tvm/build/libtvm.so | grep libcudart.so | awk '{print $1}')
  mv ${ANTARES_DRIVER_PATH}/antares_driver.cc ${ANTARES_DRIVER_PATH}/.antares_driver.cc
fi

export LD_LIBRARY_PATH=${ANTARES_DRIVER_PATH}:${LD_LIBRARY_PATH}
export HIP_PLATFORM=hcc
export HSA_USERPTR_FOR_PAGED_MEM=0

ldconfig >/dev/null 2>&1 || true
rm -f ${ANTARES_DRIVER_PATH}/property.cache

[[ "$USING_GDB" == "" ]] || USING_GDB="gdb --ex run --args"

time OP=${OP:-auto.generic} STEP=${STEP:-0} ${USING_GDB} python3 ./antares/antares_compiler.py "$@"
