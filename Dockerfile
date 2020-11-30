FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
MAINTAINER Wei CUI <weicu@microsoft.com>

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV HIP_PLATFORM hcc
ENV PATH $PATH:/opt/rocm/bin:/usr/local/nvidia/lib64/bin
ENV TVM_HOME=/opt/tvm
ENV HSA_USERPTR_FOR_PAGED_MEM=0
ENV TF_ROCM_FUSION_ENABLE 1

RUN env > /etc/environment

RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make file openssh-server kmod gdb libopenmpi-dev openmpi-bin \
        autoconf automake autotools-dev libtool llvm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] http://repo.radeon.com/rocm/apt/3.8/ xenial main" | tee /etc/apt/sources.list.d/rocm_hip.list && \
    apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rocm-dev zlib1g-dev unzip librdmacm-dev rocblas hipsparse rccl rocfft rocrand miopen-hip && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /bin/echo -e "set nocindent\nset noautoindent\nset ts=4" > /root/.vimrc

RUN pip3 install --upgrade pip cmake && \
    pip3 install --upgrade tornado psutil xgboost==1.2.1 numpy decorator attrs pytest typed_ast && \
    rm -rf ~/.cache
RUN git clone https://github.com/apache/incubator-tvm $TVM_HOME && \
    cd $TVM_HOME && git checkout 73f425d && \
    git submodule init && git submodule update && \
    mkdir -p build && cd build && cp ../cmake/config.cmake . && \
    sed -i 's/LLVM OFF/LLVM ON/g' config.cmake && sed -i 's/CUDA OFF/CUDA ON/g' config.cmake && \
    cmake .. && make -j16

ADD engine/tvm_v0.7.patch $TVM_HOME/tvm_v0.7.patch
RUN cd $TVM_HOME && git apply tvm_v0.7.patch
RUN cd $TVM_HOME && cd build && make -j16

RUN [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] || ln -s /host/usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu
RUN ln -sf libcudart.so /usr/local/cuda/targets/x86_64-linux/lib/libcudart_static.a

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV PYTHONPATH=/opt/tvm/python:/opt/tvm/topi/python:/opt/tvm/nnvm/python:/antares
ENV HIP_IGNORE_HCC_VERSION=1

