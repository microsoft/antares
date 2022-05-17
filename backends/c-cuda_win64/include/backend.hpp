// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-cuda_win64): [x86_64-w64-mingw32-g++] -O2 -static -lpthread

#include <random>
#include <sstream>

#define NVCUDA_LIBRARY_PATH R"(C:\Windows\System32\nvcuda.dll)"

#define CHECK(stat, reason, ...)  ((stat) ? 1 : (fprintf(stderr, "[CheckFail] "), fprintf(stderr, reason, ##__VA_ARGS__), fprintf(stderr, "\n"), fflush(stderr), exit(1), 0))
#define LOAD_ONCE(func, ftype)   static FARPROC __ ## func; if (!__ ## func) { __ ## func = GetProcAddress(hLibDll, #func); CHECK(__ ## func, "No such function symbol defined: %s()", #func); } auto func = (ftype)__ ## func;

namespace ab {

  static HMODULE hLibDll;
  static int _current_device;
  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  void init(int dev) {
    ab::hLibDll = LoadLibrary(NVCUDA_LIBRARY_PATH);
    CHECK(hLibDll, "Cannot find `" NVCUDA_LIBRARY_PATH "` !\n");

    void *ctx;
    LOAD_ONCE(cuInit, int (*)(int));
    LOAD_ONCE(cuDevicePrimaryCtxRetain, int (*)(void*, int));
    LOAD_ONCE(cuCtxSetCurrent, int (*)(void*));

    CHECK(0 == cuInit(0), "Failed initialize NVIDIA CUDA device with `" NVCUDA_LIBRARY_PATH "` (No NVIDIA GPU installed or enabled?).");
    CHECK(0 == cuDevicePrimaryCtxRetain(&ctx, dev), "Failed initialize NVIDIA CUDA device with `" NVCUDA_LIBRARY_PATH "` (No NVIDIA GPU installed or enabled?).");
    CHECK(0 == cuCtxSetCurrent(ctx), "Failed initialize NVIDIA CUDA device with `" NVCUDA_LIBRARY_PATH "` (No NVIDIA GPU installed or enabled?).");

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
    LOAD_ONCE(cuMemAlloc_v2, int (*)(void*, size_t));
    CHECK(0 == cuMemAlloc_v2(&dptr, byteSize), "Failed to allocate device memory.");
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  std::string moduleCompile(const std::string &source) {
    ab_utils::TempFile tempfile("cu", source);
    auto path = tempfile.get_path();

    constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    LOAD_ONCE(cuDeviceGetAttribute, int (*)(int*, int, int));
    int major, minor;
    CHECK(0 == cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _current_device), "Failed to get device attribution: CAPABILITY_MAJOR");
    CHECK(0 == cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _current_device), "Failed to get device attribution: CAPABILITY_MINOR");
    auto arch = std::to_string(major * 10 + minor);

    ab_utils::Process({"nvcc.exe", path, "--fatbin", "-gencode", "arch=compute_" + arch + ",code=sm_" + arch, "-O2", "-o", path + ".out", "1>&2"}, 10);
    return file_read((path + ".out").c_str());
  }

  void* moduleLoad(const std::string &binary) {
    void *hModule;
    LOAD_ONCE(cuModuleLoadData, int (*)(void*, const char*));
    CHECK(0 == cuModuleLoadData(&hModule, binary.c_str()), "Failed to compiler sources with command `nvcc.exe` from Windows PATH and load target to NVIDIA GPU.");
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
    LOAD_ONCE(cuModuleGetFunction, int (*)(void*, const void*, const char*));
    CHECK(0 == cuModuleGetFunction(&hFunction, hModule, fname.c_str()), "Failed to get function `%s` from module.", fname.c_str());
    return { hFunction, query("blockIdx.x"), query("blockIdx.y"), query("blockIdx.z"), query("threadIdx.x"), query("threadIdx.y"), query("threadIdx.z") };
  }

  void launchKernel(const std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    LOAD_ONCE(cuLaunchKernel, int (*)(...));
    std::vector<void* const*> pargs(krnl_args.size());
    for (int i = 0; i < pargs.size(); ++i)
      pargs[i] = &krnl_args[i];
    CHECK(0 == cuLaunchKernel(hFunc[0], (int)(size_t)hFunc[1], (int)(size_t)hFunc[2], (int)(size_t)hFunc[3], (int)(size_t)hFunc[4], (int)(size_t)hFunc[5], (int)(size_t)hFunc[6], 0, stream, (void**)pargs.data(), nullptr), "Failed to launch kernel.");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    LOAD_ONCE(cuMemcpyHtoDAsync_v2, int (*)(...));
    CHECK(0 == cuMemcpyHtoDAsync_v2(dptr, hptr, byteSize, stream), "Failed to copy memory to device.");
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    LOAD_ONCE(cuMemcpyDtoHAsync_v2, int (*)(...));
    CHECK(0 == cuMemcpyDtoHAsync_v2(hptr, dptr, byteSize, stream), "Failed to copy memory from device.");
  }

  void synchronize(void *stream) {
    LOAD_ONCE(cuStreamSynchronize, int (*)(void*));
    CHECK(0 == cuStreamSynchronize(stream), "Failed to sychronize default stream.");
  }

  void* recordTime(void *stream) {
    void *hEvent;
    LOAD_ONCE(cuEventCreate, int (*)(void*, int));
    LOAD_ONCE(cuEventRecord, int (*)(void*, void*));
    CHECK(0 == cuEventCreate(&hEvent, 0), "Failed to create event.");
    CHECK(0 == cuEventRecord(hEvent, stream), "Failed to record event.");
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    LOAD_ONCE(cuCtxSynchronize, int (*)());
    CHECK(0 == cuCtxSynchronize(), "Failed to sychronize device.");

    float ms;
    LOAD_ONCE(cuEventElapsedTime, int (*)(float*, void*, void*));
    LOAD_ONCE(cuEventDestroy, int (*)(void*));

    CHECK(0 == cuEventElapsedTime(&ms, hStart, hStop), "Failed to compute elapsed time.");
    CHECK(0 == cuEventDestroy(hStart), "Failed to destroy released event.");
    CHECK(0 == cuEventDestroy(hStop), "Failed to destroy released event.");
    return (double)ms * 1e-3;
  }
}

