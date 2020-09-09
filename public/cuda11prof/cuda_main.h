// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

CUresult cuGetErrorString(CUresult error, const char **pStr) {
  LOAD_DLSYM();
  return __func(error, pStr);
}
CUresult cuGetErrorName(CUresult error, const char **pStr) {
  LOAD_DLSYM();
  return __func(error, pStr);
}
CUresult cuInit(unsigned int Flags) {
  LOAD_DLSYM();
  return __func(Flags);
}
CUresult cuDriverGetVersion(int *driverVersion) {
  LOAD_DLSYM();
  return __func(driverVersion);
}
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  LOAD_DLSYM();
  return __func(device, ordinal);
}
CUresult cuDeviceGetCount(int *count) {
  LOAD_DLSYM();
  return __func(count);
}
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  LOAD_DLSYM();
  return __func(name, len, dev);
}
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  LOAD_DLSYM();
  return __func(uuid, dev);
}
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  LOAD_DLSYM();
  return __func(bytes, dev);
}
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
  LOAD_DLSYM();
  return __func(pi, attrib, dev);
}
CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
  LOAD_DLSYM();
  return __func(nvSciSyncAttrList, dev, flags);
}
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  LOAD_DLSYM();
  return __func(prop, dev);
}
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
  LOAD_DLSYM();
  return __func(major, minor, dev);
}
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  LOAD_DLSYM();
  return __func(pctx, dev);
}
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  LOAD_DLSYM();
  return __func(dev);
}
CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
  LOAD_DLSYM();
  return __func(dev, flags);
}
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
  LOAD_DLSYM();
  return __func(dev, flags, active);
}
CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  LOAD_DLSYM();
  return __func(dev);
}
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  LOAD_DLSYM();
  return __func(pctx, flags, dev);
}
CUresult cuCtxDestroy_v2(CUcontext ctx) {
  LOAD_DLSYM();
  return __func(ctx);
}
CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  LOAD_DLSYM();
  return __func(ctx);
}
CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  LOAD_DLSYM();
  return __func(pctx);
}
CUresult cuCtxSetCurrent(CUcontext ctx) {
  LOAD_DLSYM();
  return __func(ctx);
}
CUresult cuCtxGetCurrent(CUcontext *pctx) {
  LOAD_DLSYM();
  return __func(pctx);
}
CUresult cuCtxGetDevice(CUdevice *device) {
  LOAD_DLSYM();
  return __func(device);
}
CUresult cuCtxGetFlags(unsigned int *flags) {
  LOAD_DLSYM();
  return __func(flags);
}
CUresult cuCtxSynchronize(void) {
  LOAD_DLSYM();
  return __func();
}
CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
  LOAD_DLSYM();
  return __func(limit, value);
}
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  LOAD_DLSYM();
  return __func(pvalue, limit);
}
CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  LOAD_DLSYM();
  return __func(pconfig);
}
CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
  LOAD_DLSYM();
  return __func(config);
}
CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  LOAD_DLSYM();
  return __func(pConfig);
}
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
  LOAD_DLSYM();
  return __func(config);
}
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  LOAD_DLSYM();
  return __func(ctx, version);
}
CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  LOAD_DLSYM();
  return __func(leastPriority, greatestPriority);
}
CUresult cuCtxResetPersistingL2Cache(void) {
  LOAD_DLSYM();
  return __func();
}
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  LOAD_DLSYM();
  return __func(pctx, flags);
}
CUresult cuCtxDetach(CUcontext ctx) {
  LOAD_DLSYM();
  return __func(ctx);
}
CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  LOAD_DLSYM();
  return __func(module, fname);
}
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  LOAD_DLSYM();
  return __func(module, image);
}
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
  LOAD_DLSYM();
  return __func(module, image, numOptions, options, optionValues);
}
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
  LOAD_DLSYM();
  return __func(module, fatCubin);
}
CUresult cuModuleUnload(CUmodule hmod) {
  LOAD_DLSYM();
  return __func(hmod);
}
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  LOAD_DLSYM();
  CUresult res = __func(hfunc, hmod, name);
  if (res == CUDA_SUCCESS)
    funcNames[*hfunc] = name;
  return res;
}
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  LOAD_DLSYM();
  return __func(dptr, bytes, hmod, name);
}
CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
  LOAD_DLSYM();
  return __func(pTexRef, hmod, name);
}
CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
  LOAD_DLSYM();
  return __func(pSurfRef, hmod, name);
}
CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut) {
  LOAD_DLSYM();
  return __func(numOptions, options, optionValues, stateOut);
}
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
  LOAD_DLSYM();
  return __func(state, type, data, size, name, numOptions, options, optionValues);
}
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues) {
  LOAD_DLSYM();
  return __func(state, type, path, numOptions, options, optionValues);
}
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
  LOAD_DLSYM();
  return __func(state, cubinOut, sizeOut);
}
CUresult cuLinkDestroy(CUlinkState state) {
  LOAD_DLSYM();
  return __func(state);
}
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  LOAD_DLSYM();
  return __func(free, total);
}
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  LOAD_DLSYM();
  return __func(dptr, bytesize);
}
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
  LOAD_DLSYM();
  return __func(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}
CUresult cuMemFree_v2(CUdeviceptr dptr) {
  LOAD_DLSYM();
  return __func(dptr);
}
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
  LOAD_DLSYM();
  return __func(pbase, psize, dptr);
}
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
  LOAD_DLSYM();
  return __func(pp, bytesize);
}
CUresult cuMemFreeHost(void *p) {
  LOAD_DLSYM();
  return __func(p);
}
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(pp, bytesize, Flags);
}
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(pdptr, p, Flags);
}
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
  LOAD_DLSYM();
  return __func(pFlags, p);
}
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
  LOAD_DLSYM();
  return __func(dptr, bytesize, flags);
}
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  LOAD_DLSYM();
  return __func(dev, pciBusId);
}
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  LOAD_DLSYM();
  return __func(pciBusId, len, dev);
}
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  LOAD_DLSYM();
  return __func(pHandle, event);
}
CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
  LOAD_DLSYM();
  return __func(phEvent, handle);
}
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  LOAD_DLSYM();
  return __func(pHandle, dptr);
}
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(pdptr, handle, Flags);
}
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
  LOAD_DLSYM();
  return __func(dptr);
}
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(p, bytesize, Flags);
}
CUresult cuMemHostUnregister(void *p) {
  LOAD_DLSYM();
  return __func(p);
}
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dst, src, ByteCount);
}
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstDevice, srcHost, ByteCount);
}
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstHost, srcDevice, ByteCount);
}
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstDevice, srcDevice, ByteCount);
}
CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstArray, dstOffset, srcDevice, ByteCount);
}
CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstDevice, srcArray, srcOffset, ByteCount);
}
CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstArray, dstOffset, srcHost, ByteCount);
}
CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstHost, srcArray, srcOffset, ByteCount);
}
CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
  LOAD_DLSYM();
  return __func(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
  LOAD_DLSYM();
  return __func(pCopy);
}
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
  LOAD_DLSYM();
  return __func(pCopy);
}
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
  LOAD_DLSYM();
  return __func(pCopy);
}
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
  LOAD_DLSYM();
  return __func(pCopy);
}
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dst, src, ByteCount, hStream);
}
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, srcHost, ByteCount, hStream);
}
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstHost, srcDevice, ByteCount, hStream);
}
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, srcDevice, ByteCount, hStream);
}
CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstArray, dstOffset, srcHost, ByteCount, hStream);
}
CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstHost, srcArray, srcOffset, ByteCount, hStream);
}
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  LOAD_DLSYM();
  return __func(pCopy, hStream);
}
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
  LOAD_DLSYM();
  return __func(pCopy, hStream);
}
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
  LOAD_DLSYM();
  return __func(pCopy, hStream);
}
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  LOAD_DLSYM();
  return __func(dstDevice, uc, N);
}
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  LOAD_DLSYM();
  return __func(dstDevice, us, N);
}
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  LOAD_DLSYM();
  return __func(dstDevice, ui, N);
}
CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, uc, Width, Height);
}
CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, us, Width, Height);
}
CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, ui, Width, Height);
}
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, uc, N, hStream);
}
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, us, N, hStream);
}
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, ui, N, hStream);
}
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, uc, Width, Height, hStream);
}
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, us, Width, Height, hStream);
}
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) {
  LOAD_DLSYM();
  return __func(dstDevice, dstPitch, ui, Width, Height, hStream);
}
CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  LOAD_DLSYM();
  return __func(pHandle, pAllocateArray);
}
CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  LOAD_DLSYM();
  return __func(pArrayDescriptor, hArray);
}
CUresult cuArrayDestroy(CUarray hArray) {
  LOAD_DLSYM();
  return __func(hArray);
}
CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  LOAD_DLSYM();
  return __func(pHandle, pAllocateArray);
}
CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
  LOAD_DLSYM();
  return __func(pArrayDescriptor, hArray);
}
CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels) {
  LOAD_DLSYM();
  return __func(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}
CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) {
  LOAD_DLSYM();
  return __func(pLevelArray, hMipmappedArray, level);
}
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  LOAD_DLSYM();
  return __func(hMipmappedArray);
}
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags) {
  LOAD_DLSYM();
  return __func(ptr, size, alignment, addr, flags);
}
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  LOAD_DLSYM();
  return __func(ptr, size);
}
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags) {
  LOAD_DLSYM();
  return __func(handle, size, prop, flags);
}
CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  LOAD_DLSYM();
  return __func(handle);
}
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags) {
  LOAD_DLSYM();
  return __func(ptr, size, offset, handle, flags);
}
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  LOAD_DLSYM();
  return __func(ptr, size);
}
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count) {
  LOAD_DLSYM();
  return __func(ptr, size, desc, count);
}
CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr) {
  LOAD_DLSYM();
  return __func(flags, location, ptr);
}
CUresult cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags) {
  LOAD_DLSYM();
  return __func(shareableHandle, handle, handleType, flags);
}
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType) {
  LOAD_DLSYM();
  return __func(handle, osHandle, shHandleType);
}
CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option) {
  LOAD_DLSYM();
  return __func(granularity, prop, option);
}
CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle) {
  LOAD_DLSYM();
  return __func(prop, handle);
}
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
  LOAD_DLSYM();
  return __func(handle, addr);
}
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
  LOAD_DLSYM();
  return __func(data, attribute, ptr);
}
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) {
  LOAD_DLSYM();
  return __func(devPtr, count, dstDevice, hStream);
}
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) {
  LOAD_DLSYM();
  return __func(devPtr, count, advice, device);
}
CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) {
  LOAD_DLSYM();
  return __func(data, dataSize, attribute, devPtr, count);
}
CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) {
  LOAD_DLSYM();
  return __func(data, dataSizes, attributes, numAttributes, devPtr, count);
}
CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr) {
  LOAD_DLSYM();
  return __func(value, attribute, ptr);
}
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) {
  LOAD_DLSYM();
  return __func(numAttributes, attributes, data, ptr);
}
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(phStream, Flags);
}
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
  LOAD_DLSYM();
  return __func(phStream, flags, priority);
}
CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
  LOAD_DLSYM();
  return __func(hStream, priority);
}
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  LOAD_DLSYM();
  return __func(hStream, flags);
}
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  LOAD_DLSYM();
  return __func(hStream, pctx);
}
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(hStream, hEvent, Flags);
}
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) {
  LOAD_DLSYM();
  return __func(hStream, callback, userData, flags);
}
CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
  LOAD_DLSYM();
  return __func(hStream, mode);
}
CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  LOAD_DLSYM();
  return __func(mode);
}
CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  LOAD_DLSYM();
  return __func(hStream, phGraph);
}
CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
  LOAD_DLSYM();
  return __func(hStream, captureStatus);
}
CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus, cuuint64_t *id) {
  LOAD_DLSYM();
  return __func(hStream, captureStatus, id);
}
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) {
  LOAD_DLSYM();
  return __func(hStream, dptr, length, flags);
}
CUresult cuStreamQuery(CUstream hStream) {
  LOAD_DLSYM();
  return __func(hStream);
}
CUresult cuStreamSynchronize(CUstream hStream) {
  LOAD_DLSYM();
  return __func(hStream);
}
CUresult cuStreamDestroy_v2(CUstream hStream) {
  LOAD_DLSYM();
  return __func(hStream);
}
CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
  LOAD_DLSYM();
  return __func(dst, src);
}
CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue *value_out) {
  LOAD_DLSYM();
  return __func(hStream, attr, value_out);
}
CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr, const CUstreamAttrValue *value) {
  LOAD_DLSYM();
  return __func(hStream, attr, value);
}
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(phEvent, Flags);
}
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  LOAD_DLSYM();
  return __func(hEvent, hStream);
}
CUresult cuEventQuery(CUevent hEvent) {
  LOAD_DLSYM();
  return __func(hEvent);
}
CUresult cuEventSynchronize(CUevent hEvent) {
  LOAD_DLSYM();
  return __func(hEvent);
}
CUresult cuEventDestroy_v2(CUevent hEvent) {
  LOAD_DLSYM();
  return __func(hEvent);
}
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
  LOAD_DLSYM();
  return __func(pMilliseconds, hStart, hEnd);
}
CUresult cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
  LOAD_DLSYM();
  return __func(extMem_out, memHandleDesc);
}
CUresult cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
  LOAD_DLSYM();
  return __func(devPtr, extMem, bufferDesc);
}
CUresult cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
  LOAD_DLSYM();
  return __func(mipmap, extMem, mipmapDesc);
}
CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
  LOAD_DLSYM();
  return __func(extMem);
}
CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
  LOAD_DLSYM();
  return __func(extSem_out, semHandleDesc);
}
CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
  LOAD_DLSYM();
  return __func(extSemArray, paramsArray, numExtSems, stream);
}
CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream) {
  LOAD_DLSYM();
  return __func(extSemArray, paramsArray, numExtSems, stream);
}
CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  LOAD_DLSYM();
  return __func(extSem);
}
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
  LOAD_DLSYM();
  return __func(stream, addr, value, flags);
}
CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
  LOAD_DLSYM();
  return __func(stream, addr, value, flags);
}
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
  LOAD_DLSYM();
  return __func(stream, addr, value, flags);
}
CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) {
  LOAD_DLSYM();
  return __func(stream, addr, value, flags);
}
CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
  LOAD_DLSYM();
  return __func(stream, count, paramArray, flags);
}
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
  LOAD_DLSYM();
  return __func(pi, attrib, hfunc);
}
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
  LOAD_DLSYM();
  return __func(hfunc, attrib, value);
}
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  LOAD_DLSYM();
  return __func(hfunc, config);
}
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
  LOAD_DLSYM();
  return __func(hfunc, config);
}
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  LOAD_DLSYM();
  float ms;
  CUevent hStart, hStop;
  assert(0 == cuEventCreate(&hStart, 0));
  assert(0 == cuEventCreate(&hStop, 0));
  assert(0 == cuEventRecord(hStart, hStream));
  assert(0 == __func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra));
  assert(0 == cuEventRecord(hStop, hStream));
  assert(0 == cuEventSynchronize(hStop));
  assert(0 == cuEventElapsedTime(&ms, hStart, hStop));

  static int top_ms = -1;
  int runs = 1;
  if (top_ms < 0)
    top_ms = getenv("TOPMS") ? atoi(getenv("TOPMS")) : 0;
  if (top_ms) {
    runs = int(1000 / ms);
    runs = (runs < 1 ? 1 : runs);

    assert(0 == cuEventRecord(hStart, hStream));
    for (int i = 0; i < runs; ++i)
      assert(0 == __func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra));
    assert(0 == cuEventRecord(hStop, hStream));
    assert(0 == cuEventSynchronize(hStop));
    assert(0 == cuEventElapsedTime(&ms, hStart, hStop));
    assert(0 == cuEventDestroy(hStart));
    assert(0 == cuEventDestroy(hStop));
  }
  printf("[libnvprof11] %s -> %g ms (runs = %d)\n", funcNames[f].c_str(), ms / runs, runs);
  return CUDA_SUCCESS;
}
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) {
  LOAD_DLSYM();
  return __func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}
CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags) {
  LOAD_DLSYM();
  return __func(launchParamsList, numDevices, flags);
}
CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
  LOAD_DLSYM();
  return __func(hStream, fn, userData);
}
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  LOAD_DLSYM();
  return __func(hfunc, x, y, z);
}
CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  LOAD_DLSYM();
  return __func(hfunc, bytes);
}
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  LOAD_DLSYM();
  return __func(hfunc, numbytes);
}
CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  LOAD_DLSYM();
  return __func(hfunc, offset, value);
}
CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
  LOAD_DLSYM();
  return __func(hfunc, offset, value);
}
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
  LOAD_DLSYM();
  return __func(hfunc, offset, ptr, numbytes);
}
CUresult cuLaunch(CUfunction f) {
  LOAD_DLSYM();
  return __func(f);
}
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  LOAD_DLSYM();
  return __func(f, grid_width, grid_height);
}
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) {
  LOAD_DLSYM();
  return __func(f, grid_width, grid_height, hStream);
}
CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(hfunc, texunit, hTexRef);
}
CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
  LOAD_DLSYM();
  return __func(phGraph, flags);
}
CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}
CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
}
CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
}
CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}
CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hNode, nodeParams);
}
CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
}
CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
  LOAD_DLSYM();
  return __func(hNode, phGraph);
}
CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies) {
  LOAD_DLSYM();
  return __func(phGraphNode, hGraph, dependencies, numDependencies);
}
CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
  LOAD_DLSYM();
  return __func(phGraphClone, originalGraph);
}
CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph) {
  LOAD_DLSYM();
  return __func(phNode, hOriginalNode, hClonedGraph);
}
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
  LOAD_DLSYM();
  return __func(hNode, type);
}
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
  LOAD_DLSYM();
  return __func(hGraph, nodes, numNodes);
}
CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
  LOAD_DLSYM();
  return __func(hGraph, rootNodes, numRootNodes);
}
CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges) {
  LOAD_DLSYM();
  return __func(hGraph, from, to, numEdges);
}
CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies) {
  LOAD_DLSYM();
  return __func(hNode, dependencies, numDependencies);
}
CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes) {
  LOAD_DLSYM();
  return __func(hNode, dependentNodes, numDependentNodes);
}
CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
  LOAD_DLSYM();
  return __func(hGraph, from, to, numDependencies);
}
CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies) {
  LOAD_DLSYM();
  return __func(hGraph, from, to, numDependencies);
}
CUresult cuGraphDestroyNode(CUgraphNode hNode) {
  LOAD_DLSYM();
  return __func(hNode);
}
CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
  LOAD_DLSYM();
  return __func(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hGraphExec, hNode, nodeParams);
}
CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
  LOAD_DLSYM();
  return __func(hGraphExec, hNode, copyParams, ctx);
}
CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
  LOAD_DLSYM();
  return __func(hGraphExec, hNode, memsetParams, ctx);
}
CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams) {
  LOAD_DLSYM();
  return __func(hGraphExec, hNode, nodeParams);
}
CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  LOAD_DLSYM();
  return __func(hGraphExec, hStream);
}
CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
  LOAD_DLSYM();
  return __func(hGraphExec);
}
CUresult cuGraphDestroy(CUgraph hGraph) {
  LOAD_DLSYM();
  return __func(hGraph);
}
CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out) {
  LOAD_DLSYM();
  return __func(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}
CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
  LOAD_DLSYM();
  return __func(dst, src);
}
CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue *value_out) {
  LOAD_DLSYM();
  return __func(hNode, attr, value_out);
}
CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue *value) {
  LOAD_DLSYM();
  return __func(hNode, attr, value);
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  LOAD_DLSYM();
  return __func(numBlocks, func, blockSize, dynamicSMemSize);
}
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
  LOAD_DLSYM();
  return __func(numBlocks, func, blockSize, dynamicSMemSize, flags);
}
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) {
  LOAD_DLSYM();
  return __func(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
  LOAD_DLSYM();
  return __func(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
}
CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func, int numBlocks, int blockSize) {
  LOAD_DLSYM();
  return __func(dynamicSmemSize, func, numBlocks, blockSize);
}
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(hTexRef, hArray, Flags);
}
CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(hTexRef, hMipmappedArray, Flags);
}
CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
  LOAD_DLSYM();
  return __func(ByteOffset, hTexRef, dptr, bytes);
}
CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
  LOAD_DLSYM();
  return __func(hTexRef, desc, dptr, Pitch);
}
CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
  LOAD_DLSYM();
  return __func(hTexRef, fmt, NumPackedComponents);
}
CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
  LOAD_DLSYM();
  return __func(hTexRef, dim, am);
}
CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  LOAD_DLSYM();
  return __func(hTexRef, fm);
}
CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  LOAD_DLSYM();
  return __func(hTexRef, fm);
}
CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  LOAD_DLSYM();
  return __func(hTexRef, bias);
}
CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) {
  LOAD_DLSYM();
  return __func(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}
CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
  LOAD_DLSYM();
  return __func(hTexRef, maxAniso);
}
CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  LOAD_DLSYM();
  return __func(hTexRef, pBorderColor);
}
CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(hTexRef, Flags);
}
CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pdptr, hTexRef);
}
CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(phArray, hTexRef);
}
CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(phMipmappedArray, hTexRef);
}
CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
  LOAD_DLSYM();
  return __func(pam, hTexRef, dim);
}
CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pfm, hTexRef);
}
CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pFormat, pNumChannels, hTexRef);
}
CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pfm, hTexRef);
}
CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pbias, hTexRef);
}
CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}
CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pmaxAniso, hTexRef);
}
CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pBorderColor, hTexRef);
}
CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(pFlags, hTexRef);
}
CUresult cuTexRefCreate(CUtexref *pTexRef) {
  LOAD_DLSYM();
  return __func(pTexRef);
}
CUresult cuTexRefDestroy(CUtexref hTexRef) {
  LOAD_DLSYM();
  return __func(hTexRef);
}
CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(hSurfRef, hArray, Flags);
}
CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  LOAD_DLSYM();
  return __func(phArray, hSurfRef);
}
CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  LOAD_DLSYM();
  return __func(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}
CUresult cuTexObjectDestroy(CUtexObject texObject) {
  LOAD_DLSYM();
  return __func(texObject);
}
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
  LOAD_DLSYM();
  return __func(pResDesc, texObject);
}
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
  LOAD_DLSYM();
  return __func(pTexDesc, texObject);
}
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) {
  LOAD_DLSYM();
  return __func(pResViewDesc, texObject);
}
CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
  LOAD_DLSYM();
  return __func(pSurfObject, pResDesc);
}
CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
  LOAD_DLSYM();
  return __func(surfObject);
}
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject) {
  LOAD_DLSYM();
  return __func(pResDesc, surfObject);
}
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
  LOAD_DLSYM();
  return __func(canAccessPeer, dev, peerDev);
}
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
  LOAD_DLSYM();
  return __func(peerContext, Flags);
}
CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  LOAD_DLSYM();
  return __func(peerContext);
}
CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
  LOAD_DLSYM();
  return __func(value, attrib, srcDevice, dstDevice);
}
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  LOAD_DLSYM();
  return __func(resource);
}
CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) {
  LOAD_DLSYM();
  return __func(pArray, resource, arrayIndex, mipLevel);
}
CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) {
  LOAD_DLSYM();
  return __func(pMipmappedArray, resource);
}
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) {
  LOAD_DLSYM();
  return __func(pDevPtr, pSize, resource);
}
CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
  LOAD_DLSYM();
  return __func(resource, flags);
}
CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  LOAD_DLSYM();
  return __func(count, resources, hStream);
}
CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) {
  LOAD_DLSYM();
  return __func(count, resources, hStream);
}
CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
  LOAD_DLSYM();
  return __func(ppExportTable, pExportTableId);
}
CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
  LOAD_DLSYM();
  return __func(hmod, hfunc);
}
