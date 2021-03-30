#!/bin/bash -e

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

VERSION_TAG=v0.2dev5

REQUIRED_PACKAGES="git python3-dev python3-pip g++ llvm-dev make curl libopenmpi-dev openmpi-bin"

if grep Microsoft /proc/sys/kernel/osrelease >/dev/null; then
  REQUIRED_PACKAGES="${REQUIRED_PACKAGES} g++-mingw-w64-x86-64"
fi

if [[ "$(whoami)" != "root" ]]; then
  echo "Root previledge is required for dependency installation."
  exit 1
fi

dpkg -L ${REQUIRED_PACKAGES} >/dev/null 2>&1 || \
  sh -c "apt-get update && apt-get install -y --no-install-recommends ${REQUIRED_PACKAGES}"

TVM_HOME=/opt/tvm
GIT_COMMIT=73f425d

if [ ! -e ${TVM_HOME} ] || ! sh -c "cd ${TVM_HOME} && git fetch && git reset --hard && git checkout ${GIT_COMMIT}"; then
  rm -rf ${TVM_HOME}
  git clone https://github.com/apache/incubator-tvm ${TVM_HOME}
  sh -c "cd ${TVM_HOME} && git checkout ${GIT_COMMIT}"
fi

cd ${TVM_HOME}

python3 -m pip install --upgrade pip cmake==3.18.0 setuptools && \
  rm -rf ${TVM_HOME}/device-stub && \
  cp -r ${ANTARES_ROOT}/engine/device-stub ${TVM_HOME}/device-stub && \
  echo '' > /tmp/device-stub.c && gcc /tmp/device-stub.c -shared -o ${TVM_HOME}/device-stub/lib64/libcudart.so && \
  rm -f /tmp/device-stub.c

git checkout 73f425d && git apply device-stub/tvm_v0.7.patch && \
  git submodule init && git submodule update && \
  mkdir -p build && cd build && cp ../cmake/config.cmake . && \
  sed -i 's/LLVM OFF/LLVM ON/g' config.cmake && sed -i 's~CUDA OFF~CUDA '"${TVM_HOME}/device-stub"'~g' config.cmake && \
  cmake .. && make -j8

python3 -m pip install --upgrade tornado psutil xgboost==1.2.1 numpy decorator attrs pytest typed_ast mpi4py

echo "$VERSION_TAG" > $TVM_HOME/VERSION_TAG
