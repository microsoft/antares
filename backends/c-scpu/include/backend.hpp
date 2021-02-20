// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-scpu): -ldl

#include <dlfcn.h>
#include <pthread.h>

namespace ab {

  void init() {
  }

  void* alloc(const tensor_property &tp) {
    void *dptr = malloc(tp.mem_size());
    fprintf(stderr, "alloc(%p, `%s`);\n", dptr, tp.name.c_str());
    return dptr;
  }

  void release(void *dptr) {
    free(dptr);
    fprintf(stderr, "release(%p);\n", dptr);
  }

  void* moduleLoad(const std::string &source) {
    char temp_name[] = ".antares-module-XXXXXX";
    auto folder = std::string(mkdtemp(temp_name));
    fprintf(stderr, "load(%s)\n", folder.c_str());

    auto path = folder + "/module.cc";
    FILE *fp = fopen(path.c_str(), "w");
    assert(source.size() == fwrite(source.data(), 1, source.size(), fp));
    fclose(fp);
    assert(0 == system(("g++ " + path + " -ldl -lpthread -fPIC -shared -D__rank__=0 -O2 -o " + path + ".out").c_str()));

    void *hmod = dlopen((path + ".out").c_str(), RTLD_LAZY);

    assert(0 == system(("rm -rf " + folder).c_str()));
    return hmod;
  }

  void* moduleGetFunction(const void *hModule, const std::string &fname) {
    fprintf(stderr, "moduleGetFunction(%s)\n", fname.c_str());
    return dlsym((void*)hModule, fname.c_str());
  }

  void launchKernel(const void* hFunction, const std::unordered_map<std::string, int> &threads, const std::vector<void*> &krnl_args) {
    ((void(*)(void* const*))hFunction)(krnl_args.data());
    return;

    fprintf(stderr, "launch(");
    for (int i = 0; i < krnl_args.size(); ++i)
      fprintf(stderr, "%p,", krnl_args[i]);
    fprintf(stderr, "\b);\n");
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    fprintf(stderr, "memcpyHtoD(%zd)\n", byteSize);
    memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    fprintf(stderr, "memcpyDtoH(%zd)\n", byteSize);
    memcpy(hptr, dptr, byteSize);
  }

  void synchronize() {
    fprintf(stderr, "synchronize()\n");
  }

  void* recordTime() {
    auto pt = new std::chrono::high_resolution_clock::time_point;
    *pt = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "recordTime()\n");
    return pt;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
    auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

    fprintf(stderr, "convertToElapsedTime()\n");
    double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
    delete h1, h2;
    return et;
  }
}

