// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>
#include <cuda.h>

#include <unordered_map>
#include <string>

#if !defined(CUDA_VERSION) || CUDA_VERSION != 11000
#error "Nvprof11 only targets to CUDA 11.0 environment."
#else

static void *libcuda = NULL;
static std::unordered_map<void*, std::string> funcNames;

#ifdef __cplusplus
extern "C" {
#endif

#define LOAD_DLSYM()  \
    static CUresult (*__func)(...) = nullptr; \
    if (!__func) { \
      if (!libcuda) { \
        libcuda = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY); \
        if (libcuda == NULL) libcuda = dlopen("/usr/local/cuda/compat/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY), assert(libcuda != NULL); \
      } \
      __func = (decltype(__func))dlsym(libcuda, __func__); \
    }

#include "cuda_main.h"
////////////////////////////////////////////////////////


#ifdef __cplusplus
}
#endif

#endif
