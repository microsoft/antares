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
_STUB(cudaDeviceGetAttribute)
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
_STUB(cuLaunchKernel)
_STUB(cuMemsetD32)

_STUB(nvrtcCreateProgram)
_STUB(nvrtcGetErrorString)
_STUB(nvrtcGetProgramLogSize)
_STUB(nvrtcGetProgramLog)
_STUB(nvrtcCompileProgram)
_STUB(nvrtcGetPTXSize)
_STUB(nvrtcGetPTX)
_STUB(nvrtcDestroyProgram)

#endif
