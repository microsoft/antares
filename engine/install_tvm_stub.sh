#!/bin/sh -ex

cd $(dirname $0)/..
ANTARES_ROOT=$(pwd)
export TVM_HOME=/opt/tvm

dpkg -L llvm-dev git python3-pip >/dev/null 2>&1 || \
  apt-get update && apt-get install -y --no-install-recommends git python3-pip g++ llvm-dev && rm -rf /var/lib/apt/lists/*

pip3 install --upgrade pip cmake==3.18.0 && \
  pip3 install --upgrade tornado psutil xgboost==1.2.1 numpy decorator attrs pytest typed_ast && \
  rm -rf ~/.cache

rm -rf $TVM_HOME && git clone https://github.com/apache/incubator-tvm $TVM_HOME

cp -r ${ANTARES_ROOT}/engine/device-stub ${TVM_HOME}/device-stub && \
  echo '' > /tmp/device-stub.c && gcc /tmp/device-stub.c -shared -o ${TVM_HOME}/device-stub/lib64/libcudart.so

cd $TVM_HOME && git checkout 73f425d && git apply device-stub/tvm_v0.7.patch && \
  git submodule init && git submodule update && \
  mkdir -p build && cd build && cp ../cmake/config.cmake . && \
  sed -i 's/LLVM OFF/LLVM ON/g' config.cmake && sed -i 's~CUDA OFF~CUDA '"${TVM_HOME}/device-stub"'~g' config.cmake && \
  cmake .. && make -j8

