#!/bin/bash -e

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

bash -e ./engine/check_environ.sh

export ANTARES_DRIVER_PATH=${ANTARES_DRIVER_PATH:-${HOME}/.cache/antares}
export BACKEND=$(./antares/get_backend.sh)

if [[ "$@" == "clean" ]]; then
  set -x
  rm -rf "${ANTARES_DRIVER_PATH}"/* "${ANTARES_DRIVER_PATH}"/.??*
  exit 0
elif [[ "$@" == "rest-server" ]]; then
  export HTTP_SERVICE=1
  shift
elif [[ "$@" == "check-backend" ]]; then
  echo $BACKEND
  exit 0
elif [[ "$@" == "backends" ]]; then
  ls -1 ./backends | grep -v c-base
  exit 0
elif [[ "$@" == "help" ]]; then
  if which less >/dev/null; then
    exec less ./README.md
  else
    exec more ./README.md
  fi
fi

if grep Microsoft /proc/sys/kernel/osrelease >/dev/null || grep WSL2 /proc/sys/kernel/osrelease >/dev/null; then
  export IS_WSL=1
else
  export IS_WSL=0
fi

if [ -e ${ANTARES_ROOT}/3rdparty/tvm ]; then
  export TVM_HOME=${ANTARES_ROOT}/3rdparty/tvm
else
  export TVM_HOME=${HOME}/.local/antares/3rdparty/tvm
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${TVM_HOME}/python:${TVM_HOME}/topi/python:${TVM_HOME}/nnvm/python:${ANTARES_ROOT}:${PYTHONPATH}:${WORKDIR}

VERSION_TAG=$(cat engine/install_antares_host.sh | grep ^VERSION_TAG | head -n 1 | awk -F\= '{print $NF}')

if [[ "$(cat ${TVM_HOME}/VERSION_TAG 2>/dev/null)" != "${VERSION_TAG}" ]] || [ ! -e ${TVM_HOME}/build/libtvm.so ]; then
  echo 'Antares dependencies are not up-to-date/fully installed for current Antares version. Try updating the dependencies with: `make install_host` ..'
  exit 1
fi

if [[ "$COMPUTE_V1" == "" ]]; then
  export COMPUTE_V1='- einstein_v2("output0[N, M] = input0[N, M] + input1[N, M]", input_dict={"input0": {"dtype": "float32", "shape": [1024, 512]}, "input1": {"dtype": "float32", "shape": [1024, 512]}})'
fi

mkdir -p ${ANTARES_DRIVER_PATH}

if [[ "$BACKEND" == "c-rocm" ]]; then
  /opt/rocm/bin/hipcc engine/cuda_properties.cc -o ${ANTARES_DRIVER_PATH}/device_properties >/dev/null 2>&1 || true
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg 2>/dev/null || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
elif [[ "$BACKEND" == "c-cuda" ]]; then
  g++ engine/cuda_properties.cc -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64 -o ${ANTARES_DRIVER_PATH}/device_properties >/dev/null 2>&1 || true
  ${ANTARES_DRIVER_PATH}/device_properties > ${ANTARES_DRIVER_PATH}/device_properties.cfg 2>/dev/null || rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
else
  rm -f ${ANTARES_DRIVER_PATH}/device_properties.cfg
fi

if [ ! -e ${ANTARES_DRIVER_PATH}/device_properties.cfg ]; then
  if [[ "${HARDWARE_CONFIG}" != "" ]]; then
    cat hardware/${HARDWARE_CONFIG}.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    # echo "  >> Using specific hardware device properties."
  elif [ -e backends/${BACKEND}/default_props.cfg ]; then
    cat backends/${BACKEND}/default_props.cfg > ${ANTARES_DRIVER_PATH}/device_properties.cfg
    # echo "  >> Using ${BACKEND} default device properties."
  else
    echo -e "\n  >> Unsupported Backend: No device properties found for backend type: ${BACKEND}.\n"
    echo -e   "  >> Valid Backend Types include:\n"
    echo -e   "        " $(ls ./backends -1 | grep "^c-*")
    echo
    exit 1
  fi
else
  true # echo "  >> Using ${BACKEND} runtime device properties."
fi


ldconfig >/dev/null 2>&1 || true

[[ "$USING_GDB" == "" ]] || USING_GDB="gdb --ex run --args"

STEP=${STEP:-0} ${USING_GDB} exec ${PYTHON_EXEC:-python3} ./antares/antares_compiler.py "$@"
