#!/bin/bash -e

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)

VERSION_TAG=v0.3dev1

if [[ "$NO_PYTHON" != "1" ]]; then

bash -e ./engine/check_environ.sh

REQUIRED_CMDS="git python3 g++ make"

if grep Microsoft /proc/sys/kernel/osrelease >/dev/null || grep WSL2 /proc/sys/kernel/osrelease >/dev/null; then
  REQUIRED_CMDS="${REQUIRED_CMDS} x86_64-w64-mingw32-c++"
fi

for CMD in ${REQUIRED_CMDS}; do
  if ! which ${CMD} >/dev/null; then
    echo
    echo "[Error] Command '${CMD}' not found in user PATH. Please install this package to satisfy each antares dependency in: ${REQUIRED_CMDS} (from 'g++-mingw-w64-x86-64')"
    echo
    exit 1
  fi
done

fi

set -x

TVM_HOME=${HOME}/.local/antares/3rdparty/tvm
GIT_COMMIT=v0.8.0

if [ ! -e ${TVM_HOME} ] || ! sh -ce "cd ${TVM_HOME}; git fetch; git reset --hard; git checkout ${GIT_COMMIT}"; then
  rm -rf ${TVM_HOME}
  mkdir -p ${TVM_HOME}
  git clone https://github.com/apache/tvm ${TVM_HOME}
  sh -c "cd ${TVM_HOME} && git checkout ${GIT_COMMIT}"
fi

cd ${TVM_HOME}

python3 -m pip install --user --upgrade pip cmake==3.18.0 setuptools

rm -rf ${TVM_HOME}/device-stub
cp -r ${ANTARES_ROOT}/engine/device-stub ${TVM_HOME}/device-stub
touch ${TVM_HOME}/device-stub/device-stub.c && gcc ${TVM_HOME}/device-stub/device-stub.c -shared -o ${TVM_HOME}/device-stub/lib64/libcudart.so

git checkout ${GIT_COMMIT}
git apply device-stub/tvm_v0.8.0.patch
git apply device-stub/tvm_extra.patch
echo 'register_func("tvm_callback_cuda_compile", lambda code: bytearray(), override=True)' >> python/tvm/__init__.py

git submodule init && git submodule update
mkdir -p build && cd build && cp ../cmake/config.cmake .

sed -i 's/LLVM OFF/LLVM OFF/g' config.cmake && sed -i 's~CUDA OFF~CUDA '"${TVM_HOME}/device-stub"'~g' config.cmake
echo 'set(USE_THREADS OFF)' >> config.cmake

PATH="${HOME}/.local/bin:${PATH}" cmake ..
make -j$(nproc)

if [[ "$NO_PYTHON" != "1" ]]; then
  python3 -m pip install --user --upgrade tornado psutil xgboost==1.2.1 numpy decorator attrs pytest typed_ast cloudpickle scipy
fi

echo "$VERSION_TAG" > $TVM_HOME/VERSION_TAG
