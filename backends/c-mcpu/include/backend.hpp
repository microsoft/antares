// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-mcpu): -ldl

#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>

namespace ab {

  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;

  void init(int dev) {
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
    void *dptr = memalign(sysconf(_SC_PAGESIZE), byteSize);
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    auto &it = _cached_memory[byteSize];
    it.push_back(dptr);
  }

  void* moduleLoad(const std::string &source) {
    ab_utils::TempFile tempfile("cpp", source);
    auto path = tempfile.get_path();

    ab_utils::Process({"g++", path, "-std=c++17", "-ldl", "-lpthread", "-fPIC", "-shared", "-O2", "-o", path + ".out", "-ffast-math", "-march=native"}, 10);

    path = (path[0] == '/' ? path : "./" + path) + ".out";
    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    return { dlsym((void*)hModule, fname.c_str()), (void*)(long)threads.find("__rank__")->second };
  }

  static std::vector<std::vector<void*>> _task_queue;
  static long _max_threads_in_task_queue;
  static pthread_barrier_t _thread_barrier;

  static void* thread_worker(void* args) {
    long rank = (long)args;
    for (auto &it: _task_queue) {
      auto func = ((void(*)(int, void* const*))it[0]);
      if (rank < (long)it[1])
        func(rank, it.data() + 2);
      pthread_barrier_wait(&_thread_barrier);
    }
    return nullptr;
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args, void *stream) {
    std::vector<void*> task = hFunction;
    task.insert(task.end(), krnl_args.begin(), krnl_args.end());
    _task_queue.push_back(std::move(task));

    _max_threads_in_task_queue = std::max(_max_threads_in_task_queue, (long)hFunction[1]);
  }

  void synchronize(void *stream) {
    if (_task_queue.size()) {
      std::vector<pthread_t> tid(_max_threads_in_task_queue);
      pthread_barrier_init(&_thread_barrier, nullptr, tid.size());

      for (int i = 0; i < tid.size(); ++i)
        pthread_create(&tid[i], nullptr, thread_worker, (void*)(long)i);
      for (int i = 0; i < tid.size(); ++i)
        pthread_join(tid[i], nullptr);

      pthread_barrier_destroy(&_thread_barrier);
      _max_threads_in_task_queue = 0;
      _task_queue.clear();
    }
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);

    memcpy(dptr, hptr, byteSize);
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    ab::synchronize(stream);

    memcpy(hptr, dptr, byteSize);
  }

  void* recordTime(void *stream) {
    ab::synchronize(stream);

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
