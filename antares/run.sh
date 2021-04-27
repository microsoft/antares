#!/bin/bash -e

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

export ANTARES_DRIVER_PATH=${ANTARES_ROOT}/.libAntares

if [[ "$@" == "clean" ]]; then
  rm -rf "${ANTARES_DRIVER_PATH}"
  exit 0
fi

export PYTHONDONTWRITEBYTECODE=1
export TVM_HOME=${HOME}/.local/antares/thirdparty/tvm
export PYTHONPATH=${TVM_HOME}/python:${TVM_HOME}/topi/python:${TVM_HOME}/nnvm/python:${ANTARES_ROOT}

VERSION_TAG=$(cat engine/install_antares_host.sh | grep ^VERSION_TAG | head -n 1 | awk -F\= '{print $NF}')

if grep Microsoft /proc/sys/kernel/osrelease >/dev/null; then
  export IS_WSL=1
else
  export IS_WSL=0
fi

if [[ "$(cat ${TVM_HOME}/VERSION_TAG 2>/dev/null)" != "${VERSION_TAG}" ]]; then
  if [[ "$IS_WSL" == "1" ]]; then
    echo 'Antares dependencies are not up-to-date with current Antares version. Try updating the dependencies with: `make install_host` ..'
    make install_host
  else
    echo 'Antares dependencies are not up-to-date with current Antares version. Please update the dependencies with: `make install_host`'
    exit 1
  fi
elif [ ! -e ${TVM_HOME}/build/libtvm.so ]; then
  if [[ "$IS_WSL" == "1" ]]; then
    echo 'Antares dependencies are not fully installed in this environment. Try installing them with: `make install_host`'
    make install_host
  else
    echo 'Antares dependencies are not fully installed in this environment. Plese install them with: `make install_host`'
    exit 1
  fi
fi

if [[ "$COMPUTE_V1" == "" ]]; then
  export COMPUTE_V1='- einstein_v2("output0[N, M] = input0[N, M] + input1[N, M]", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [1024, 512]}})'
  echo "  >> Using a default computing expression for testing."
fi

export BACKEND=$(./antares/get_backend.sh)

mkdir -p ${ANTARES_DRIVER_PATH}

if [[ "$BACKEND" == "c-rocm" ]]; then
  /opt/rocm/bin/hipcc engine/cuda_properties.cc -o ${ANTARES_DRIVER_PATH}/device_properties >/dev/null 2>&1 || true
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg 2>/dev/null || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
elif [[ "$BACKEND" == "c-cuda" ]]; then
  g++ engine/cuda_properties.cc -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o ${ANTARES_DRIVER_PATH}/device_properties >/dev/null 2>&1 || true
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg 2>/dev/null || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
else
  rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
fi

if [ ! -e ${ANTARES_DRIVER_PATH}/device_properties.cfg ]; then
  if [[ "${HARDWARE_CONFIG}" != "" ]]; then
    cat hardware/${HARDWARE_CONFIG}.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    echo "  >> Using specific hardware device properties."
  elif [ -e backends/${BACKEND}/default_props.cfg ]; then
    cat backends/${BACKEND}/default_props.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    echo "  >> Using ${BACKEND} default device properties."
  else
    echo -e "\n  >> Unsupported Backend: No device properties found for backend type: ${BACKEND}.\n"
    exit 1
  fi
else
  echo "  >> Using ${BACKEND} runtime device properties."
fi


ldconfig >/dev/null 2>&1 || true

[[ "$USING_GDB" == "" ]] || USING_GDB="gdb --ex run --args"

time STEP=${STEP:-0} ${USING_GDB} python3 ./antares/antares_compiler.py "$@"
