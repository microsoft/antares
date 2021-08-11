// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-rocm_win64): [x86_64-w64-mingw32-g++] -O2 -static -lpthread

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <random>
#include <sstream>

#define AMDHIP64_LIBRARY_PATH R"(C:\Windows\System32\amdhip64.dll)"

#define CHECK(stat, reason, ...)  ((stat) ? 1 : (fprintf(stderr, "[CheckFail] "), fprintf(stderr, reason, ##__VA_ARGS__), fprintf(stderr, "\n"), fflush(stderr), exit(1), 0))
#define LOAD_ONCE(func, ftype)   static FARPROC __ ## func; if (!__ ## func) { __ ## func = GetProcAddress(hLibDll, #func); CHECK(__ ## func, "No such function symbol defined: %s()", #func); } auto func = (ftype)__ ## func;

namespace ab {

  static HMODULE hLibDll;
  static int _current_device;
  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  void init(int dev) {
    ab::hLibDll = LoadLibrary(AMDHIP64_LIBRARY_PATH);
    CHECK(hLibDll, "Cannot find `" AMDHIP64_LIBRARY_PATH "` !\n");

    LOAD_ONCE(hipSetDevice, int (*)(int));
    CHECK(0 == hipSetDevice(dev), "Failed initialize AMD ROCm device with `" AMDHIP64_LIBRARY_PATH "` (No AMDGPU installed or enabled?).");
    _current_device = dev;
  }

  void finalize() {
    if (ab::hLibDll != nullptr)
      FreeLibrary(ab::hLibDll), ab::hLibDll = nullptr;
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    auto &it = _cached_memory[byteSize];
    if (it.size()) {
      auto dptr = it.back();
      it.pop_back();
      return dptr;
    }
    void *dptr = nullptr;
    LOAD_ONCE(hipMalloc, int (*)(void*, size_t));
    CHECK(0 == hipMalloc(&dptr, byteSize), "Failed to allocate device memory.");
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    ab_utils::TempFile tempfile("cu", source);
    auto path = tempfile.get_path();

    ab_utils::Process({"wsl.exe", "sh", "-cx", "\"/opt/rocm/bin/hipcc " + path + " --amdgpu-target=gfx803 --amdgpu-target=gfx900 --amdgpu-target=gfx906 --amdgpu-target=gfx908 --amdgpu-target=gfx1010 --genco -Wno-ignored-attributes -O2 -o " + path + ".out 1>&2\""}, 10);

    void *hModule;
    LOAD_ONCE(hipModuleLoad, int (*)(void*, const char*));
    CHECK(0 == hipModuleLoad(&hModule, (path + ".out").c_str()), "Command `/opt/rocm/bin/hipcc` in WSL is not found, please install latest `rocm-dev` package in WSL environment for C-code compilation to AMD's HSACO.");
    return hModule;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    auto query = [&](const std::string &axis, size_t defval = 1) -> void* {
      auto it = threads.find(axis);
      if (it == threads.end())
        return (void*)defval;
      return (void*)(size_t)it->second;
    };

    void *hFunction;
    LOAD_ONCE(hipModuleGetFunction, int (*)(void*, const void*, const char*));
    CHECK(0 == hipModuleGetFunction(&hFunction, hModule, fname.c_str()), "Failed to get function `%s` from ROCm HSACO module.", fname.c_str());
    return { hFunction, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z") };
  }

  void launchKernel(const std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    LOAD_ONCE(hipModuleLaunchKernel, int (*)(...));
    std::vector<void* const*> pargs(krnl_args.size());
    for (int i = 0; i < pargs.size(); ++i)
      pargs[i] = &krnl_args[i];
    CHECK(0 == hipModuleLaunchKernel(hFunc[0], (int)(size_t)hFunc[1], (int)(size_t)hFunc[2], (int)(size_t)hFunc[3], (int)(size_t)hFunc[4], (int)(size_t)hFunc[5], (int)(size_t)hFunc[6], 0, stream, (void**)pargs.data(), nullptr), "Failed to launch kernel.");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    LOAD_ONCE(hipMemcpyHtoDAsync, int (*)(...));
    CHECK(0 == hipMemcpyHtoDAsync(dptr, hptr, byteSize, stream), "Failed to copy memory to device.");
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    LOAD_ONCE(hipMemcpyDtoHAsync, int (*)(...));
    CHECK(0 == hipMemcpyDtoHAsync(hptr, dptr, byteSize, stream), "Failed to copy memory from device.");
  }

  void synchronize(void *stream) {
    LOAD_ONCE(hipStreamSynchronize, int (*)(void*));
    CHECK(0 == hipStreamSynchronize(stream), "Failed to sychronize default stream.");
  }

  void* recordTime(void *stream) {
    void *hEvent;
    LOAD_ONCE(hipEventCreate, int (*)(void*, int));
    LOAD_ONCE(hipEventRecord, int (*)(void*, void*));
    CHECK(0 == hipEventCreate(&hEvent, 0), "Failed to create event.");
    CHECK(0 == hipEventRecord(hEvent, stream), "Failed to record event.");
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    LOAD_ONCE(hipDeviceSynchronize, int (*)());
    CHECK(0 == hipDeviceSynchronize(), "Failed to sychronize device.");

    float ms;
    LOAD_ONCE(hipEventElapsedTime, int (*)(float*, void*, void*));
    LOAD_ONCE(hipEventDestroy, int (*)(void*));

    CHECK(0 == hipEventElapsedTime(&ms, hStart, hStop), "Failed to compute elapsed time.");
    CHECK(0 == hipEventDestroy(hStart), "Failed to destroy released event.");
    CHECK(0 == hipEventDestroy(hStop), "Failed to destroy released event.");
    return (double)ms * 1e-3;
  }
}

