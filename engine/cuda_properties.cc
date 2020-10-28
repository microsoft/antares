// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#if !defined(__HIPCC__)
#include <cuda_runtime_api.h>
#define Q(attr_key) ((0 == cudaDeviceGetAttribute(&val, cudaDevAttr ## attr_key, 0)) ? printf("%s: %d\n", #attr_key, val) : (exit(1), 0))
#define CHECK_ENV() assert(getenv("BACKEND") != NULL), assert(strcmp(getenv("BACKEND"), "c-cuda") == 0);
#else
#include <hip/hip_runtime.h>
#define Q(attr_key) ((0 == hipDeviceGetAttribute(&val, hipDeviceAttribute ## attr_key, 0)) ? printf("%s: %d\n", #attr_key, val) : (exit(1), 0))
#define hipDeviceAttributeMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define CHECK_ENV() assert(getenv("BACKEND") != NULL), assert(strcmp(getenv("BACKEND"), "c-rocm") == 0);
#endif

int main() {
	int val = -1;
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
	return 0;
}
