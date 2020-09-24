// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define Q(attr_key) assert(0 == cudaDeviceGetAttribute(&val, attr_key, 0)), printf("cuda-attr %s(%d) = %d\n", #attr_key, attr_key, val)

int main() {
	int val = -1;
	Q(cudaDevAttrMaxThreadsPerBlock);
	Q(cudaDevAttrWarpSize);
	Q(cudaDevAttrMaxSharedMemoryPerBlock);
	Q(cudaDevAttrComputeCapabilityMajor);
	Q(cudaDevAttrComputeCapabilityMinor);
	Q(cudaDevAttrClockRate);
	Q(cudaDevAttrMultiProcessorCount);
	Q(cudaDevAttrMaxBlockDimX);
	Q(cudaDevAttrMaxBlockDimY);
	Q(cudaDevAttrMaxBlockDimZ);
	Q(cudaDevAttrMaxRegistersPerBlock);
	return 0;
}
