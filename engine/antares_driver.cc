// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <assert.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <execinfo.h>
#include <malloc.h>
#include <sys/mman.h>
#include <hip/hip_runtime.h>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <deque>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

#define LOGGING_API()  if (verbose) fprintf(stdout, "<<%u/%p>> call %s\n", getpid(), (void*)pthread_self(), __func__);
#define ERROR()        (printf("error in %s..\n", __func__), _exit(1))

#define cudaDeviceAttr int
#define CUDA_SUCCESS hipSuccess
#define CUresult hipError_t
#define CUDAAPI
#define CUDARTAPI CUDAAPI
#define CUdeviceptr hipDeviceptr_t
#define CUmodule hipModule_t
#define CUfunction hipFunction_t
#define CUstream hipStream_t

#define cudaError_t CUresult
#define cudaSuccess CUDA_SUCCESS
#define cudaEvent_t hipEvent_t
#define cudaStream_t CUstream
#define __device_builtin__


#define ktrace() { \
    void *array[100]; \
    size_t size; \
    char **strings; \
    size_t i; \
    size = backtrace(array, 100); \
    strings = backtrace_symbols(array, size); \
    if (NULL == strings) { \
        perror("backtrace_symbols"); \
        exit(EXIT_FAILURE); \
    } \
    printf(" - Obtained %zd stack frames.\n", size); \
    for (i = 0; i < size; i++) \
        printf("    # %s\n", strings[i]); \
    free(strings); \
    strings = NULL; \
}

enum __device_builtin__ cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,
    cudaMemcpyHostToDevice        =   1,
    cudaMemcpyDeviceToHost        =   2,
    cudaMemcpyDeviceToDevice      =   3,
    cudaMemcpyDefault             =   4
};

static int verbose = -1, no_device = 0;
static void *libaccel = NULL;
static const char *backend = NULL;

#define LOAD_DLSYM(fROCm, fCUDA)  \
    static CUresult (*__l)(...); \
    if (!libaccel) { \
      if (0 == strcmp(backend, "c-rocm")) { \
        libaccel = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LOCAL | RTLD_LAZY), assert(libaccel != NULL); \
        printf("  >> HIP runtime Loaded successfully for pid = %u.\n", getpid()); \
      } else if (0 == strcmp(backend, "c-cuda")) { \
        libaccel = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY); \
        if (libaccel == NULL) libaccel = dlopen("/usr/local/cuda/compat/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY), assert(libaccel != NULL); \
        printf("  >> CUDA runtime Loaded successfully for pid = %u.\n", getpid()); \
      } else { \
        printf("  >> [Error] No valid drivers found for backend: %s.\n", backend), _exit(1); \
      } \
    } \
    if (!__l) __l = (decltype(__l))dlsym(libaccel, (0 == strcmp(backend, "c-rocm")) ? #fROCm : #fCUDA);


class CudartInitializor {

public:
  CudartInitializor() {
    verbose = getenv("V") ? atoi(getenv("V")) : 0;
    backend = getenv("BACKEND");
    assert(strlen(backend) >= 3);
  }

  ~CudartInitializor() {
  }
};

static CudartInitializor __init__;

extern "C" {

CUresult CUDAAPI cudaDeviceGetAttribute(int *ri, cudaDeviceAttr prop, int device) {
  LOGGING_API();
  printf("  >> Unrecognized property value = %u\n", prop);
  assert(0);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, int dev) {
  snprintf(name, len, "Device:%u/Type:%s", dev, backend);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr) {
  static char str[128];
  snprintf(str, sizeof(str), "Driver error code = %u\n", error);
  *pStr = str;
  return CUDA_SUCCESS;
}

const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
  static char str[128];
  snprintf(str, sizeof(str), "Runtime error code = %u\n", error);
  return str;
}

CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  LOGGING_API();
  assert(0);
  return CUDA_SUCCESS;
}

cudaError_t CUDARTAPI cudaGetDevice(int *device) {
  LOGGING_API();
  *device = 0;
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
  LOGGING_API();
  assert(device == 0);
  if (no_device)
    printf("  >> No %s device available.\n", backend), _exit(1);

  if (backend[2] == 'c') {
    static bool once = true;
    if (!once)
      return cudaSuccess;
    do {
      once = false;
      { LOAD_DLSYM(_, cuInit); if (0 != __l(0)) break; }
      void *pctx;
      { LOAD_DLSYM(_, cuDevicePrimaryCtxRetain); if (0 != __l(&pctx, 0)) break; }
      { LOAD_DLSYM(_, cuCtxSetCurrent); if (0 != __l(pctx)) break; }
      return cudaSuccess;
    } while (1);
    printf("  >> No CUDA device available.\n"), _exit(1);
  }

  LOAD_DLSYM(hipSetDevice, _);
  if (0 != __l(device))
    printf("  >> No ROCm device available.\n"), _exit(1);
  return cudaSuccess;
}

//////////////////////////////////////////////////////////

cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
  LOGGING_API();
  assert(0);
  return cudaSuccess;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  LOGGING_API();

  switch (kind) {
    case cudaMemcpyHostToDevice: {
      LOAD_DLSYM(hipMemcpyHtoDAsync, cuMemcpyHtoDAsync_v2);
      assert(0 == __l(dst, (void*)src, count, stream));
      return cudaSuccess;
    }
    case cudaMemcpyDeviceToHost: {
      LOAD_DLSYM(hipMemcpyDtoHAsync, cuMemcpyDtoHAsync_v2);
      assert(0 == __l(dst, (void*)src, count, stream));
      return cudaSuccess;
    }
    default:
      assert(0);
  }
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  LOGGING_API();
  assert(0 == cudaMemcpyAsync(dst, src, count, kind, 0));
  return cudaStreamSynchronize(0);
}

CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  LOGGING_API();
  LOAD_DLSYM(hipMemsetD32, cuMemsetD32_v2);
  assert(0 == __l(dstDevice, ui, N));
  return CUDA_SUCCESS;
}

cudaError_t CUDARTAPI cudaMallocHost(void **hostPtr, size_t size) {
  LOGGING_API();
  LOAD_DLSYM(hipMallocHost, cuMemAllocHost_v2);
  assert(0 == __l(hostPtr, size));
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFreeHost(void **hostPtr) {
  LOGGING_API();
  LOAD_DLSYM(hipFreeHost, cuMemFreeHost);
  assert(0 == __l(hostPtr));
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  LOGGING_API();
  LOAD_DLSYM(hipMalloc, cuMemAlloc_v2);
  assert(0 == __l(devPtr, size));
  return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFree(void *devPtr) {
  LOGGING_API();
  LOAD_DLSYM(hipFree, cuMemFree_v2);
  assert(0 == __l(devPtr));
  return cudaSuccess;
}

////////////////////////////////////////////////////////

CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const char *image)  {
  LOGGING_API();
  LOAD_DLSYM(hipModuleLoadData, cuModuleLoadData);
  assert(0 == __l(module, image));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod)  {
  LOGGING_API();
  LOAD_DLSYM(hipModuleUnload, cuModuleUnload);
  assert(0 == __l(hmod));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  LOGGING_API();
  LOAD_DLSYM(hipModuleGetFunction, cuModuleGetFunction);
  assert(0 == __l(hfunc, hmod, name));
  return CUDA_SUCCESS;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  LOGGING_API();
  LOAD_DLSYM(hipStreamSynchronize, cuStreamSynchronize);
  assert(0 == __l(stream));
  return cudaSuccess;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  LOGGING_API();
  LOAD_DLSYM(hipModuleLaunchKernel, cuLaunchKernel);
  assert(0 == __l(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                    sharedMemBytes, hStream, kernelParams, nullptr));
  return CUDA_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////
} // extern "C"

