#!/bin/bash -e

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

export PYTHONDONTWRITEBYTECODE=1
export TVM_HOME=$(cat engine/install_antares_host.sh | grep ^TVM_HOME | head -n 1 | awk -F\= '{print $NF}')
export PYTHONPATH=${TVM_HOME}/python:${TVM_HOME}/topi/python:${TVM_HOME}/nnvm/python:${ANTARES_ROOT}

if [ ! -e ${TVM_HOME}/build/libtvm.so ]; then
  echo 'Antares dependencies are not fully installed in this environment. Try installing with: `make install_host`'
  exit 1
fi

if [[ "$COMPUTE_V1" == "" ]]; then
  echo "  >> Using a default computing expression."
  export COMPUTE_V1='- einstein_v2("output0[N] = input0[N] + input1[N]", input_dict={"input0": {"dtype": "float32", "shape": [1024 * 512]}, "input1": {"dtype": "float32", "shape": [1024 * 512]}})'
fi

export BACKEND=$(./antares/get_backend.sh)

if [[ "$(id -u)" == "0" ]]; then
  export ANTARES_DRIVER_PATH=${ANTARES_ROOT}/.tmp/libAntares
else
  export ANTARES_DRIVER_PATH=/tmp/libAntares
fi

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
  else
    cat backends/${BACKEND}/default_props.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    echo "  >> Using ${BACKEND} default device properties."
  fi
else
  echo "  >> Using ${BACKEND} runtime device properties."
fi


export HIP_PLATFORM=hcc
export HSA_USERPTR_FOR_PAGED_MEM=0

ldconfig >/dev/null 2>&1 || true

[[ "$USING_GDB" == "" ]] || USING_GDB="gdb --ex run --args"

time STEP=${STEP:-0} ${USING_GDB} python3 ./antares/antares_compiler.py "$@"
