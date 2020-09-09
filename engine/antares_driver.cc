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

enum __device_builtin__ cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  /**< Maximum number of threads per block */
    cudaDevAttrMaxBlockDimX                   = 2,  /**< Maximum block dimension X */
    cudaDevAttrMaxBlockDimY                   = 3,  /**< Maximum block dimension Y */
    cudaDevAttrMaxBlockDimZ                   = 4,  /**< Maximum block dimension Z */
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  /**< Maximum shared memory available per block in bytes */
    cudaDevAttrWarpSize                       = 10, /**< Warp size in threads */
    cudaDevAttrClockRate                      = 13, /**< Peak clock frequency in kilohertz */
    cudaDevAttrMultiProcessorCount            = 16, /**< Number of multiprocessors on device */
    cudaDevAttrComputeCapabilityMajor         = 75, /**< Major compute capability version number */
    cudaDevAttrComputeCapabilityMinor         = 76, /**< Minor compute capability version number */
    cudaDevAttrMax
};

enum class BackendType {
  C_ROCM,
  C_CUDA,
  C_MCPU,
  C_GC,
  C_HLSL,
  ANY_BUILTIN,
};

static BackendType backend_type;
static int verbose = -1, no_device = 0;
static void *libaccel = NULL;
static const char *backend = NULL;

#define LOAD_DLSYM(fROCm, fCUDA)  \
    static CUresult (*__l)(...); \
    if (!libaccel) { \
      if (backend_type == BackendType::C_ROCM) { \
        libaccel = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LOCAL | RTLD_LAZY), assert(libaccel != NULL); \
        printf("  >> HIP runtime Loaded successfully for pid = %u.\n", getpid()); \
      } else if (backend_type == BackendType::C_CUDA) { \
        libaccel = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY); \
        if (libaccel == NULL) libaccel = dlopen("/usr/local/cuda/compat/libcuda.so.1", RTLD_LOCAL | RTLD_LAZY), assert(libaccel != NULL); \
        printf("  >> CUDA runtime Loaded successfully for pid = %u.\n", getpid()); \
      } else { \
        printf("  >> [Error] No valid drivers found for backend: %s.\n", backend), _exit(1); \
      } \
    } \
    if (!__l) __l = (decltype(__l))dlsym(libaccel, (backend_type == BackendType::C_ROCM) ? #fROCm : #fCUDA);

static int attr[cudaDevAttrMax];

class CudartInitializor {

public:
  CudartInitializor() {
    verbose = getenv("V") ? atoi(getenv("V")) : 0;
    backend = getenv("BACKEND");
    auto config = getenv("HARDWARE_CONFIG");
    if (config && !*config)
      unsetenv("HARDWARE_CONFIG");

    if (!strcmp(backend, "c-rocm")) {
      backend_type = BackendType::C_ROCM;
      // setenv("HARDWARE_CONFIG", "AMD-MI50", 0);
    } else if (!strcmp(backend, "c-mcpu")) {
      backend_type = BackendType::C_MCPU;
      setenv("HARDWARE_CONFIG", "GENERIC-CPU", 0);
    } else if (!strcmp(backend, "c-gc")) {
      backend_type = BackendType::C_GC;
      setenv("HARDWARE_CONFIG", "GRAPH-CORE", 0);
    } else if (!strcmp(backend, "c-hlsl")) {
      backend_type = BackendType::C_HLSL;
      setenv("HARDWARE_CONFIG", "DX12-HLSL", 0);
    } else if (!strcmp(backend, "c-cuda")) {
      backend_type = BackendType::C_CUDA;
      // setenv("HARDWARE_CONFIG", "NVIDIA-V100", 0);
    } else {
      backend_type = BackendType::ANY_BUILTIN;
      auto config = getenv("HARDWARE_CONFIG");
      if (config == nullptr || !*config)
        printf("  >> [Error] HARDWARE_CONFIG is also needed for any unknown backend type: %s\n", backend), _exit(1);
    }
    loadAttributeValues();
  }

  ~CudartInitializor() {
  }

  void read_from_config(const char *config) {
    auto conf = "./hardware/" + std::string(config) + ".cfg";
    FILE *fp = fopen(conf.c_str(), "r");
    if (fp == NULL)
      printf("  >> [Error] HARDWARE_CONFIG file at `%s` is not found.\n", conf.c_str()), _exit(1);

    static char line[1024];
    while (fgets(line, sizeof(line), fp)) {
      char *pos = strstr(line, ": ");
      if (!pos)
        continue;
      *pos = 0;
      int val = -1;
      if (pos[2] == '$') {
        if (!strcmp(pos + 2, "$CPU_NPROC\n"))
          val = sysconf(_SC_NPROCESSORS_ONLN);
        else
          assert(0);
      } else
        val = atoi(pos + 2);

      if (!strcmp(line, "MaxThreadsPerBlock"))
        attr[cudaDevAttrMaxThreadsPerBlock] = val;
      else if (!strcmp(line, "MaxBlockDimX"))
        attr[cudaDevAttrMaxBlockDimX] = val;
      else if (!strcmp(line, "MaxBlockDimY"))
        attr[cudaDevAttrMaxBlockDimY] = val;
      else if (!strcmp(line, "MaxBlockDimZ"))
        attr[cudaDevAttrMaxBlockDimZ] = val;
      else if (!strcmp(line, "MaxSharedMemoryPerBlock"))
        attr[cudaDevAttrMaxSharedMemoryPerBlock] = val;
      else if (!strcmp(line, "WarpSize"))
        attr[cudaDevAttrWarpSize] = val;
      else if (!strcmp(line, "ClockRate"))
        attr[cudaDevAttrClockRate] = val;
      else if (!strcmp(line, "MultiProcessorCount"))
        attr[cudaDevAttrMultiProcessorCount] = val;
      else if (!strcmp(line, "ComputeCapabilityMajor"))
        attr[cudaDevAttrComputeCapabilityMajor] = val;
      else if (!strcmp(line, "ComputeCapabilityMinor"))
        attr[cudaDevAttrComputeCapabilityMinor] = val;
      else
        assert(0);
    }
    fclose(fp);
    printf("  >> Using HARDWARE_CONFIG from file: %s;\n", config);
  }

  void loadAttributeValues() {
    int prop_map[][3] = {
      {cudaDevAttrMaxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, 1024},
      {cudaDevAttrWarpSize, hipDeviceAttributeWarpSize, 64},
      {cudaDevAttrMaxSharedMemoryPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock, 64 << 10},
      {cudaDevAttrComputeCapabilityMajor, hipDeviceAttributeComputeCapabilityMajor, 9},
      {cudaDevAttrComputeCapabilityMinor, hipDeviceAttributeComputeCapabilityMinor, 6},
      {cudaDevAttrClockRate, hipDeviceAttributeClockRate, 1802000}, // 1080Ti: 1835000; Vega20: 1802000;
      {cudaDevAttrMultiProcessorCount, hipDeviceAttributeMultiprocessorCount, 60}, // 1080Ti: 20; Vega20: 60;
      {cudaDevAttrMaxBlockDimX, hipDeviceAttributeMaxBlockDimX, 1024},
      {cudaDevAttrMaxBlockDimY, hipDeviceAttributeMaxBlockDimY, 1024},
      {cudaDevAttrMaxBlockDimZ, hipDeviceAttributeMaxBlockDimZ, 64},
    };
    memset(attr, -1, sizeof(attr));
    auto config = getenv("HARDWARE_CONFIG");
    if (config != nullptr && *config) {
      read_from_config(config);
      return;
    } else {
      auto on_load_fail = [&]() {
        if (backend_type == BackendType::C_ROCM)
          read_from_config("AMD-MI50"), no_device = 1;
        else
          read_from_config("NVIDIA-V100"), no_device = 1;
      };

      std::string propertyCache = getenv("ANTARES_DRIVER_PATH") + std::string("/property.cache");

      FILE *fp = fopen(propertyCache.c_str(), "r");
      if (fp == NULL) {
        pid_t pid = fork();
        if (pid == 0) {
          { LOAD_DLSYM(hipInit, cuInit); if (0 != __l(0)) exit(0); }
          LOAD_DLSYM(hipDeviceGetAttribute, cuDeviceGetAttribute);
          fp = fopen(propertyCache.c_str(), "w"), assert(fp != NULL);
          for (int i = 0; i < sizeof(prop_map) / sizeof(*prop_map); ++i) {
            int val = -1;
            if (0 != __l(&val, prop_map[i][backend_type == BackendType::C_ROCM], 0))
              break;
            fprintf(fp, "%d\n", val);
          }
          fclose(fp);
          exit(0);
        }
        int status;
        assert(pid == waitpid(pid, &status, 0));
        fp = fopen(propertyCache.c_str(), "rb");
        if (fp == NULL)
          on_load_fail();
      }
      if (fp != NULL) {
        for (int i = 0; i < sizeof(prop_map) / sizeof(*prop_map); ++i) {
          if (1 != fscanf(fp, "%d", &attr[prop_map[i][0]])) {
            on_load_fail();
            break;
          }
        }
        fclose(fp);
      }
    }

    if (verbose) {
      for (int i = 0; i < sizeof(prop_map) / sizeof(*prop_map); ++i)
        printf("  >> Property loaded for type(%u): %d;\n", prop_map[i][0], attr[prop_map[i][0]]);
    }
  }
};

static CudartInitializor __init__;

extern "C" {

CUresult CUDAAPI cudaDeviceGetAttribute(int *ri, cudaDeviceAttr prop, int device) {
  LOGGING_API();
  if (attr[prop] == -1) {
    printf("  >> Unrecognized property value = %u\n", prop);
    abort();
  }
  *ri = attr[prop];
  return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, int dev) {
  snprintf(name, len, "Device-%u/Typeid-%u", dev, (unsigned)backend_type);
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

  if (backend_type == BackendType::C_CUDA) {
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

  assert(blockDimX * blockDimY * blockDimZ <= attr[cudaDevAttrMaxThreadsPerBlock]);

  LOAD_DLSYM(hipModuleLaunchKernel, cuLaunchKernel);
  assert(0 == __l(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                    sharedMemBytes, hStream, kernelParams, nullptr));
  return CUDA_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////
} // extern "C"

