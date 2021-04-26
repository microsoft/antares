#!/bin/bash -xe

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

VERSION_TAG=v0.2dev8

REQUIRED_PACKAGES="git python3-dev python3-pip g++ llvm-dev make curl libopenmpi-dev openmpi-bin"

if grep Microsoft /proc/sys/kernel/osrelease >/dev/null; then
  REQUIRED_PACKAGES="${REQUIRED_PACKAGES} g++-mingw-w64-x86-64"
fi

dpkg -L ${REQUIRED_PACKAGES} >/dev/null 2>&1 || \
  sudo sh -c "sudo apt-get update && sudo apt-get install -y --no-install-recommends ${REQUIRED_PACKAGES}"


TVM_HOME=${HOME}/.local/antares/thirdparty/tvm
GIT_COMMIT=0b24cbf1be

if [ ! -e ${TVM_HOME} ] || ! sh -c "cd ${TVM_HOME} && git fetch && git reset --hard && git checkout ${GIT_COMMIT}"; then
  rm -rf ${TVM_HOME}
  mkdir -p ${TVM_HOME}
  git clone https://github.com/apache/tvm ${TVM_HOME}
  sh -c "cd ${TVM_HOME} && git checkout ${GIT_COMMIT}"
fi

cd ${TVM_HOME}

python3 -m pip install --user --upgrade pip cmake==3.18.0 setuptools && \
  rm -rf ${TVM_HOME}/device-stub && \
  cp -r ${ANTARES_ROOT}/engine/device-stub ${TVM_HOME}/device-stub && \
  echo '' > ${TVM_HOME}/device-stub/device-stub.c && gcc ${TVM_HOME}/device-stub/device-stub.c -shared -o ${TVM_HOME}/device-stub/lib64/libcudart.so

git checkout ${GIT_COMMIT} && git apply device-stub/tvm_v0.7.patch && \
  git submodule init && git submodule update && \
  mkdir -p build && cd build && cp ../cmake/config.cmake . && \
  sed -i 's/LLVM OFF/LLVM ON/g' config.cmake && sed -i 's~CUDA OFF~CUDA '"${TVM_HOME}/device-stub"'~g' config.cmake && \
  PATH="${HOME}/.local/bin:${PATH}" cmake .. && make -j8

python3 -m pip install --user --upgrade tornado psutil xgboost==1.2.1 numpy decorator attrs pytest typed_ast cloudpickle

echo "$VERSION_TAG" > $TVM_HOME/VERSION_TAG
