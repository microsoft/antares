FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
MAINTAINER Wei CUI <weicu@microsoft.com>

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV HIP_PLATFORM hcc
ENV PATH $PATH:/opt/rocm/bin:/usr/local/nvidia/lib64/bin
ENV HSA_USERPTR_FOR_PAGED_MEM=0
ENV TF_ROCM_FUSION_ENABLE 1

RUN env > /etc/environment

RUN apt-get update && apt install -y --no-install-recommends git ca-certificates \
    python3-pip python3-wheel python3-setuptools python3-dev python3-pytest \
    vim-tiny less netcat-openbsd inetutils-ping curl patch iproute2 \
    g++ libpci3 libnuma-dev make file openssh-server kmod gdb libopenmpi-dev openmpi-bin psmisc \
        autoconf automake autotools-dev libtool llvm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    printf "deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.0/ xenial main" | tee /etc/apt/sources.list.d/rocm_hip.list && \
    apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rocm-dev zlib1g-dev rename zip unzip librdmacm-dev rocblas hipsparse rccl rocfft rocrand miopen-hip rocthrust hip-rocclr && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /bin/echo -e "set backspace=indent,eol,start\nset nocompatible\nset ts=4" > /etc/vim/vimrc.tiny

RUN [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] || ln -s /host/usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu
RUN ln -sf libcudart.so /usr/local/cuda/targets/x86_64-linux/lib/libcudart_static.a

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV HIP_IGNORE_HCC_VERSION=1

ADD ./engine /antares/engine
RUN /antares/engine/install_antares_host.sh && rm -rf /var/lib/apt/lists/* ~/.cache

