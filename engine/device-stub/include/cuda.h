// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef __ANTARES_CUDA_STUB__
#define __ANTARES_CUDA_STUB__

#include <assert.h>

typedef void* cudaEvent_t;
typedef void* cudaStream_t;
typedef void* CUfunction;
typedef void* CUmodule;
typedef void* CUstream;
typedef void* nvrtcProgram;

typedef unsigned long long CUdeviceptr;
typedef int cudaMemcpyKind;
typedef int cudaError_t;
typedef int CUresult;
typedef int nvrtcResult;
typedef int cudaDeviceAttr;

#define CUDA_VERSION 10000
#define CUDA_SUCCESS 0
#define CUDA_ERROR_DEINITIALIZED 4
#define cudaSuccess 0
#define cudaErrorCudartUnloading 29
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyDeviceToDevice 3
#define NVRTC_SUCCESS 0

#define cudaDevAttrMaxThreadsPerBlock 1
#define cudaDevAttrWarpSize 10
#define cudaDevAttrComputeCapabilityMajor 75
#define cudaDevAttrComputeCapabilityMinor 76
#define cudaDevAttrMaxSharedMemoryPerBlock 8
#define cudaDevAttrClockRate 13
#define cudaDevAttrMultiProcessorCount 16
#define cudaDevAttrMaxBlockDimX 2
#define cudaDevAttrMaxBlockDimY 3
#define cudaDevAttrMaxBlockDimZ 4
#define cudaDevAttrMaxRegistersPerBlock 12

#define _STUB(x)  static int x(...) { assert(#x == NULL); }

_STUB(cudaGetErrorString)
_STUB(cudaSetDevice)
_STUB(cudaGetDevice)
_STUB(cuDeviceGetName)
_STUB(cuGetErrorName)
_STUB(cuModuleGetGlobal)
_STUB(cuModuleLoadData)
_STUB(cuModuleUnload)
_STUB(cuModuleGetFunction)
_STUB(cudaMalloc)
_STUB(cudaFree)
_STUB(cudaMallocHost)
_STUB(cudaFreeHost)
_STUB(cudaMemcpyPeerAsync)
_STUB(cudaMemcpyAsync)
_STUB(cudaMemcpy)
_STUB(cudaStreamCreate)
_STUB(cudaStreamDestroy)
_STUB(cudaStreamWaitEvent)
_STUB(cudaStreamSynchronize)
_STUB(cudaEventCreate)
_STUB(cudaEventRecord)
_STUB(cudaEventDestroy)
_STUB(cudaEventSynchronize)
_STUB(cudaEventElapsedTime)
_STUB(cuLaunchKernel)
_STUB(cuMemsetD32)
_STUB(cudaMemGetInfo)

_STUB(nvrtcCreateProgram)
_STUB(nvrtcGetErrorString)
_STUB(nvrtcGetProgramLogSize)
_STUB(nvrtcGetProgramLog)
_STUB(nvrtcCompileProgram)
_STUB(nvrtcGetPTXSize)
_STUB(nvrtcGetPTX)
_STUB(nvrtcDestroyProgram)

#include <unordered_map>
#include <fstream>
#define DEF_ATTR(key_attr)  attr2sattr[cudaDevAttr ## key_attr] = # key_attr

inline cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
  static std::unordered_map<std::string, int> sattr2val;
  static std::unordered_map<cudaDeviceAttr, std::string> attr2sattr;
  if (!attr2sattr.size()) {
    std::ifstream fin(getenv("ANTARES_DRIVER_PATH") + std::string("/device_properties.cfg"));
    std::string key, val;
    while (getline(fin, key, ':') && getline(fin, val))
      sattr2val[key] = std::atoi(val.c_str());
    DEF_ATTR(MaxThreadsPerBlock);
    DEF_ATTR(WarpSize);
    DEF_ATTR(MaxSharedMemoryPerBlock);
    DEF_ATTR(ComputeCapabilityMajor);
    DEF_ATTR(ComputeCapabilityMinor);
    DEF_ATTR(ClockRate);
    DEF_ATTR(MultiProcessorCount);
    DEF_ATTR(MaxBlockDimX);
    DEF_ATTR(MaxBlockDimY);
    DEF_ATTR(MaxBlockDimZ);
    DEF_ATTR(MaxRegistersPerBlock);
    if (!sattr2val["MaxRegistersPerBlock"])
      sattr2val["MaxRegistersPerBlock"] = 64 << 10;
  }
  auto sattr = attr2sattr.find(attr);
  assert(sattr != attr2sattr.end());
  auto pvalue = sattr2val.find(sattr->second);
  assert(pvalue != sattr2val.end());
  *value = pvalue->second;
  return cudaSuccess;
}

#endif
