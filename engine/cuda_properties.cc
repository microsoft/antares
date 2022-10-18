// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if !defined(__HIPCC__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#define Q(attr_key) ((0 == cuDeviceGetAttribute(&val, (CUdevice_attribute_enum)cudaDevAttr ## attr_key, dev)) ? printf("%s: %d\n", #attr_key, val) : (exit(1), 0))
#define CHECK_ENV() (0 == cuInit(0) || (exit(1), 0));
#else
#include <hip/hip_runtime.h>
#define Q(attr_key) ((0 == hipDeviceGetAttribute_(&val, hipDeviceAttribute ## attr_key, dev)) ? printf("%s: %d\n", #attr_key, val) : (exit(1), 0))
#define hipDeviceAttributeMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define hipDeviceAttributeGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#define CHECK_ENV() (0 == hipInit(0) || (exit(1), 0));

inline hipError_t hipDeviceGetAttribute_(int *val, hipDeviceAttribute_t attr, int dev) {
  if (attr == hipDeviceAttributeComputeCapabilityMajor || attr == hipDeviceAttributeComputeCapabilityMinor) {
    static hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev);
    if (err != 0)
      return err;
    *val = (attr == hipDeviceAttributeComputeCapabilityMajor) ? (prop.gcnArch / 100) : (prop.gcnArch % 100);
    return hipSuccess;
  }
  return ::hipDeviceGetAttribute(val, attr, dev);
}

#endif

int main() {
  int val = -1, dev = getenv("DEVICE_ID") ? atoi(getenv("DEVICE_ID")) : 0;
  CHECK_ENV();

  Q(MaxThreadsPerBlock);
  Q(WarpSize);
  Q(MaxSharedMemoryPerBlock);
  Q(ComputeCapabilityMajor);
  Q(ComputeCapabilityMinor);
  Q(ClockRate);
  Q(MultiProcessorCount);
  Q(MaxBlockDimX);
  Q(MaxBlockDimY);
  Q(MaxBlockDimZ);
  Q(GlobalMemoryBusWidth);
  Q(MemoryClockRate);
  return 0;
}
