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

  void* moduleLoad(const std::string &source) {
    char temp_name[] = ".antares-module-XXXXXX";
    auto folder = std::string(mkdtemp(temp_name));
    // fprintf(stderr, "load(%s)\n", folder.c_str());

    auto path = folder + "/module.cc";
    FILE *fp = fopen(path.c_str(), "w");
    CHECK_OK(source.size() == fwrite(source.data(), 1, source.size(), fp));
    fclose(fp);
    CHECK_OK(0 == system(("timeout 10s g++ " + path + " -ldl -lpthread -fPIC -shared -O2 -o " + path + ".out").c_str()));

    void *hmod = dlopen((path + ".out").c_str(), RTLD_LAZY);
    CHECK_OK(0 == system(("rm -rf " + folder).c_str()));
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    // fprintf(stderr, "moduleGetFunction(%s)\n", fname.c_str());
    return { dlsym((void*)hModule, fname.c_str()) };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args) {
    // fprintf(stderr, "launchKernel()\n");
    ((void(*)(int, void* const*))hFunction[0])(0, krnl_args.data());
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    // fprintf(stderr, "memcpyHtoD(%zd)\n", byteSize);
    memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    // fprintf(stderr, "memcpyDtoH(%zd)\n", byteSize);
    memcpy(hptr, dptr, byteSize);
  }

  void synchronize() {
    // fprintf(stderr, "synchronize()\n");
  }

  void* recordTime() {
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
    delete h1, h2;
    return std::max(et, 1e-9);
  }
}

