// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-hlsl_win64): [x86_64-w64-mingw32-g++] -O2 -static -lpthread

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <chrono>

#define HLSL_LIBRARY_PATH R"(.\antares_hlsl_v0.3_x64.dll)"

#define CHECK(stat, reason, ...)  ((stat) ? 1 : (fprintf(stderr, "[CheckFail] "), fprintf(stderr, reason, ##__VA_ARGS__), fprintf(stderr, "\n\n"), fflush(stderr), exit(1), 0))
#define LOAD_ONCE(func, ftype)   static FARPROC __ ## func; if (!__ ## func) { __ ## func = GetProcAddress(hLibDll, #func); CHECK(__ ## func, "No such function symbol defined: %s()", #func); } auto func = (ftype)__ ## func;

namespace ab {

  static HMODULE hLibDll;
  static bool cpu_timing = false;

  void init(int dev) {
    ab::hLibDll = LoadLibrary(HLSL_LIBRARY_PATH);
    CHECK(hLibDll, "Failed to load `" HLSL_LIBRARY_PATH "`, please download these libraries first!\n");

    LOAD_ONCE(dxInit, int (*)(int));
    CHECK(0 == dxInit(0), "Failed initialize DirectX12 device.");

    LOAD_ONCE(dxEventCreate, void* (*)());
    LOAD_ONCE(dxEventRecord, int (*)(void*, void*));
    LOAD_ONCE(dxEventElapsedSecond, float (*)(void*, void*));
    LOAD_ONCE(dxEventDestroy, int (*)(void*));

    void *hEvent = dxEventCreate();
    dxEventRecord(hEvent, nullptr);
    float _t = dxEventElapsedSecond(hEvent, hEvent);
    if (_t < 0) {
      ab::cpu_timing = true;
      fprintf(stderr, "[DirectX 12] Failed to enable device timing due to improper TDR setting, falling back to cpu timing.\n"), fflush(stderr);
    }
  }

  void finalize() {
    if (ab::hLibDll != nullptr) {
      LOAD_ONCE(dxFinalize, int (*)());

      CHECK(0 == dxFinalize(), "Failed to finalize DirectX12 device.");
      FreeLibrary(ab::hLibDll), ab::hLibDll = nullptr;
    }
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

  std::string moduleCompile(const std::string &source) {
    return source;
  }

  void* moduleLoad(const std::string &binary) {
    LOAD_ONCE(dxModuleLoad, void* (*)(const char*));
    void *hModule = dxModuleLoad(binary.c_str());
    CHECK(hModule != nullptr, "Failed to load device module.");
    return hModule;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    LOAD_ONCE(dxModuleGetShader, void* (*)(const void*, const char*));
    void *hFunction = dxModuleGetShader(hModule, fname.c_str());
    CHECK(hFunction != nullptr, "Failed to get function `%s` from module.", fname.c_str());

    auto query = [&](const std::string &axis, ssize_t defval = 1) -> void* {
      auto it = threads.find(axis);
      if (it == threads.end())
        return (void*)defval;
      return (void*)(ssize_t)it->second;
    };

    std::vector<void*> fdata = { hFunction };

    void *item = query("$", 0);
    if (item) {
      fdata.push_back(item);

      for (int i = 0; ; ++i) {
        void *item = query("$" + std::to_string(i), 0);
        if (!item)
          break;
        fdata.push_back(item);
      }
    }
    return fdata;
  }

  void launchKernel(const std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    LOAD_ONCE(dxShaderLaunchAsyncExt, int (*)(...));
    ssize_t attrs = -1;
    if (hFunc.size() > 1) {
      for (int i = 2; i < hFunc.size(); ++i) {
        ssize_t val = (ssize_t)hFunc[i];
        if (val < 0) continue;

        auto ptr = (ssize_t*)&krnl_args[i - 2 + (ssize_t)hFunc[1]];
        attrs *= (*ptr + val - 1) / val;
      }
      attrs = -attrs;
    }
    CHECK(0 == dxShaderLaunchAsyncExt(hFunc[0], krnl_args.data(), krnl_args.size(), attrs, stream), "Failed to launch a shader.");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    LOAD_ONCE(dxMemcpyHtoDAsync, int (*)(void* dst, void* src, size_t bytes, void* hStream));
    CHECK(0 == dxMemcpyHtoDAsync(dptr, hptr, byteSize, stream), "Failed to copy memory to device.");
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    LOAD_ONCE(dxMemcpyDtoHAsync, int (*)(void* dst, void* src, size_t bytes, void* hStream));
    CHECK(0 == dxMemcpyDtoHAsync(hptr, dptr, byteSize, stream), "Failed to copy memory from device.");
  }

  void synchronize(void *stream) {
    LOAD_ONCE(dxStreamSynchronize, int (*)(void* hStream));
    CHECK(0 == dxStreamSynchronize(stream), "Failed to sychronize default device stream.");
  }

  void* recordTime(void *stream) {
    if (cpu_timing) {
      ab::synchronize(stream);

      auto pt = new std::chrono::high_resolution_clock::time_point;
      *pt = std::chrono::high_resolution_clock::now();
      return pt;
    }
    LOAD_ONCE(dxEventCreate, void* (*)());
    LOAD_ONCE(dxEventRecord, int (*)(void*, void*));
    void *hEvent = dxEventCreate();
    CHECK(0 == dxEventRecord(hEvent, stream), "Failed to record event to default stream.");
    return hEvent;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    if (cpu_timing) {
      auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
      auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

      double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
      delete h1, h2;
      return std::max(et, 1e-9);
    }
    ab::synchronize(nullptr);
    LOAD_ONCE(dxEventElapsedSecond, float (*)(void*, void*));
    LOAD_ONCE(dxEventDestroy, int (*)(void*));

    float sec = dxEventElapsedSecond(hStart, hStop);
    CHECK(0 == dxEventDestroy(hStart), "Failed to destroy released event.");
    CHECK(0 == dxEventDestroy(hStop), "Failed to destroy released event.");
    return (double)sec;
  }
}

