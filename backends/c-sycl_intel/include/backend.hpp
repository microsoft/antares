// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-sycl_intel): [dpcpp] -ldl
//; eval_flags(c-sycl_cuda): [/usr/local/dpcpp-cuda/bin/clang++] -ldl -I/usr/local/dpcpp-cuda/include/sycl -L/usr/local/dpcpp-cuda/lib -lsycl -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -fsycl-unnamed-lambda

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>

namespace ab {

  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;
  static sycl::queue _sycl_queue;

  void init(int dev) {
    try {
       _sycl_queue = std::move(sycl::queue(sycl::default_selector{}));
    } catch (sycl::exception const &e) {
      std::terminate();
    }
    fprintf(stderr, "   (SYCL_INFO: SYCL Device Name = %s)\n", _sycl_queue.get_device().get_info<sycl::info::device::name>().c_str());
  }

  void finalize() {
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    auto &it = _cached_memory[byteSize];
    if (it.size()) {
      auto dptr = it.back();
      it.pop_back();
      return dptr;
    }
    return memalign(sysconf(_SC_PAGESIZE), byteSize);
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    if (__BACKEND__ == "c-sycl_cuda")
      ab_utils::Process({"/usr/local/dpcpp-cuda/bin/clang++", path, "-std=c++17", "-ldl", "-fPIC", "-shared", "-O2", "-I/usr/local/dpcpp-cuda/include/sycl", "-L/usr/local/dpcpp-cuda/lib", "-lsycl", "-fsycl", "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice", "-fsycl-unnamed-lambda", "-Wno-unknown-cuda-version", "-o", path + ".out"}, 10);
    else
      ab_utils::Process({"dpcpp", path, "-std=c++17", "-lpthread", "-fPIC", "-shared", "-O2", "-o", path + ".out"}, 10);

    path = (path[0] == '/' ? path : "./" + path) + ".out";
    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    return { dlsym((void*)hModule, fname.c_str()) };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args) {
    ((void(*)(void*, void* const*))hFunction[0])(&_sycl_queue, krnl_args.data());
  }

  void synchronize() {
    _sycl_queue.wait();
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    ab::synchronize();

    memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    ab::synchronize();

    memcpy(hptr, dptr, byteSize);
  }

  void* recordTime() {
    ab::synchronize();

    auto pt = new std::chrono::high_resolution_clock::time_point;
    *pt = std::chrono::high_resolution_clock::now();
    return pt;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
    auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

    double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
    delete h1, h2;
    return std::max(et, 1e-9);
  }
}
