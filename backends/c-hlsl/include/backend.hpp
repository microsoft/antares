// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-hlsl): [x86_64-w64-mingw32-g++] -O2 -static

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define HLSL_LIBRARY_PATH R"(.\antares_hlsl_v0.2dev0_x64.dll)"

#define CHECK(stat, reason, ...)  ((stat) ? 1 : (fprintf(stderr, "[DX12:DBG] "), fprintf(stderr, reason, ##__VA_ARGS__), fprintf(stderr, "\n\n"), fflush(stderr), exit(1), 0))
#define LOAD_ONCE(func, ftype)   static FARPROC __ ## func; if (!__ ## func) { __ ## func = GetProcAddress(hLibDll, #func); CHECK(__ ## func, "No such function symbol defined: %s()", #func); } auto func = (ftype)__ ## func;

namespace ab {

  static HMODULE hLibDll;

  void init(int dev) {
    ab::hLibDll = LoadLibrary(HLSL_LIBRARY_PATH);
    CHECK(hLibDll, "Cannot find `" HLSL_LIBRARY_PATH "` !\n");

    LOAD_ONCE(dxInit, int (*)(int));
    CHECK(0 == dxInit(0), "Failed initialize DirectX12 device.");
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    LOAD_ONCE(dxMemAlloc, void* (*)(size_t bytes));
    void *dptr = dxMemAlloc(byteSize);
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    LOAD_ONCE(dxMemFree, int (*)(void* dptr));
    CHECK(0 == dxMemFree(dptr), "Failed to free device pointer.");
  }

  void* moduleLoad(const std::string &source) {
    LOAD_ONCE(dxModuleLoad, void* (*)(const char*));
    void *hModule = dxModuleLoad(source.c_str());
    CHECK(hModule != nullptr, "Failed to load device module.");
    return hModule;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    LOAD_ONCE(dxModuleGetShader, void* (*)(const void*, const char*));
    void *hFunction = dxModuleGetShader(hModule, fname.c_str());
    CHECK(hFunction != nullptr, "Failed to get function `%s` from module.", fname.c_str());
    return { hFunction };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args) {
    LOAD_ONCE(dxShaderLaunchAsync, int (*)(void*, void* const*, void*));
    CHECK(0 == dxShaderLaunchAsync(hFunction[0], krnl_args.data(), nullptr), "Failed to launch a shader.");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    LOAD_ONCE(dxMemcpyHtoDAsync, int (*)(void* dst, void* src, size_t bytes, void* hStream));
    CHECK(0 == dxMemcpyHtoDAsync(dptr, hptr, byteSize, nullptr), "Failed to copy memory to device.");
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    LOAD_ONCE(dxMemcpyDtoHAsync, int (*)(void* dst, void* src, size_t bytes, void* hStream));
    CHECK(0 == dxMemcpyDtoHAsync(hptr, dptr, byteSize, nullptr), "Failed to copy memory from device.");
  }

  void synchronize() {
    LOAD_ONCE(dxStreamSynchronize, int (*)(void* hStream));
    CHECK(0 == dxStreamSynchronize(nullptr), "Failed to sychronize default device stream.");
  }

  void* recordTime() {
    LOAD_ONCE(dxEventCreate, void* (*)());
    LOAD_ONCE(dxEventRecord, int (*)(void*, void*));
    void *hEvent = dxEventCreate();
    CHECK(0 == dxEventRecord(hEvent, nullptr), "Failed to record event to default stream.");
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    ab::synchronize();
    LOAD_ONCE(dxEventElapsedSecond, float (*)(void*, void*));
    LOAD_ONCE(dxEventDestroy, int (*)(void*));

    float sec = dxEventElapsedSecond(hStart, hStop);
    CHECK(0 == dxEventDestroy(hStart), "Failed to destroy released event.");
    CHECK(0 == dxEventDestroy(hStop), "Failed to destroy released event.");
    return (double)sec;
  }
}

