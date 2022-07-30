// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-scpu): -ldl

#include <dlfcn.h>
#include <pthread.h>

namespace ab {

  void init(int dev) {
  }

  void finalize() {
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    void *dptr = malloc(byteSize);
    // fprintf(stderr, "alloc(%p);\n", dptr);
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    // fprintf(stderr, "release(%p);\n", dptr);
    free(dptr);
  }

  std::string moduleCompile(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    ab_utils::Process({"g++", path, "-std=c++17", "-ldl", "-lpthread", "-fPIC", "-shared", "-O2", "-o", path + ".out", "-ffast-math", "-march=native"}, 10);

    path = (path[0] == '/' ? path : "./" + path) + ".out";
    return file_read(path.c_str());
  }

  void* moduleLoad(const std::string &binary) {
    ab_utils::TempFile tempfile("so", binary, false);
    auto path = tempfile.get_path();
    path = (path[0] == '/' ? path : "./" + path);

    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    // fprintf(stderr, "moduleGetFunction(%s)\n", fname.c_str());
    return { dlsym((void*)hModule, fname.c_str()), (void*)(long)threads.find("__rank__")->second };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args, void *stream) {
    // fprintf(stderr, "launchKernel()\n");
    const int num_threads = (long)hFunction[1];
    const auto func = (void(*)(int, void* const*))hFunction[0];
    for (int i = 0; i < num_threads; ++i)
      func(i, krnl_args.data());
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    // fprintf(stderr, "memcpyHtoD(%zd)\n", byteSize);
    memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    // fprintf(stderr, "memcpyDtoH(%zd)\n", byteSize);
    memcpy(hptr, dptr, byteSize);
  }

  void synchronize(void *stream) {
    // fprintf(stderr, "synchronize()\n");
  }

  void* recordTime(void *stream) {
    // fprintf(stderr, "recordTime()\n");
    auto pt = new std::chrono::high_resolution_clock::time_point;
    *pt = std::chrono::high_resolution_clock::now();
    return pt;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    // fprintf(stderr, "convertToElapsedTime()\n");
    auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
    auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

    double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
    delete h1; delete h2;
    return std::max(et, 1e-9);
  }
}

