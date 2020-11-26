// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#define _API_WRAPPER_V2_

#ifdef _API_WRAPPER_V2_
#define __EXPORT__ extern "C" __declspec(dllexport)

__EXPORT__ int dxInit(int flags);

__EXPORT__ void* dxAllocateBuffer(size_t bytes);

__EXPORT__ void dxReleaseBuffer(void* dptr);

__EXPORT__ void dxGetShaderArgumentProperty(void* handle, int arg_index, size_t* num_elements, size_t* type_size, const char** dtype_name);

__EXPORT__ void* dxCreateShader(const char* _source, int* num_inputs, int* num_outputs);

__EXPORT__ void dxDestroyShader(void* shader);

__EXPORT__ void* dxCreateStream();

__EXPORT__ void dxDestroyStream(void* stream);

__EXPORT__ void dxSubmitStream(void* stream);

__EXPORT__ void dxSynchronize(void* stream);

__EXPORT__ void dxMemcpyHostToDeviceSync(void* dst, void* src, size_t bytes);

__EXPORT__ void dxMemcpyDeviceToHostSync(void* dst, void* src, size_t bytes);

__EXPORT__ void dxLaunchShaderAsync(void* handle, void** buffers, void* stream);

__EXPORT__ void* dxCreateQuery();

__EXPORT__ void dxDestroyQuery(void* query);

__EXPORT__ void dxRecordQuery(void* query, void* stream);

__EXPORT__ double dxQueryElapsedTime(void* queryStart, void* queryEnd);
#endif