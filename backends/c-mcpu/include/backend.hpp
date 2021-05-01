// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-mcpu): -ldl -lpthread
//; eval_flags(c-mcpu_avx512): -ldl -lpthread

#include <dlfcn.h>
#include <pthread.h>
#include <malloc.h>

#ifndef __ANTARES_THREAD_POOL__
#define __ANTARES_THREAD_POOL__

#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    ThreadPool(size_t threads): stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            CHECK_OK(rc == 0);
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {

        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared< std::packaged_task<return_type()> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    const std::vector<std::thread>& get_workers() const {
      return this->workers;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

#endif

namespace ab {

  static std::unordered_map<size_t, std::vector<void*>> _cached_memory;
  static bool use_avx512;

  static std::shared_ptr<ThreadPool> _thread_pool;
  static std::vector<std::vector<void*>> _task_queue;

  void init(int dev) {
    use_avx512 = (__BACKEND__ == "c-mcpu_avx512");
    const auto env = getenv("CPU_THREADS");

    int max_allowed_threads = std::thread::hardware_concurrency();
    if (env && *env)
      max_allowed_threads = atoi(env);

    // fprintf(stderr, "[DEBUG] `max_allowed_threads` = %d\n", max_allowed_threads);
    _thread_pool = std::make_shared<ThreadPool>(max_allowed_threads);

  }

  void finalize() {
    _thread_pool = nullptr;
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

    if (use_avx512)
      ab_utils::Process({"clang++-10", path, "-std=c++17", "-ldl", "-lpthread", "-fPIC", "-shared", "-O3", "-o", path + ".out", "-ffast-math", "-march=skylake-avx512"}, 10);
    else
      ab_utils::Process({"g++", path, "-std=c++17", "-ldl", "-lpthread", "-fPIC", "-shared", "-O3", "-o", path + ".out", "-ffast-math", "-march=native"}, 10);

    path = (path[0] == '/' ? path : "./" + path) + ".out";
    void* hmod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    CHECK_OK(hmod != nullptr);
    return hmod;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    return { dlsym((void*)hModule, fname.c_str()), (void*)(long)threads.find("__rank__")->second };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args, void *stream) {
    std::vector<void*> task = hFunction;
    task.insert(task.end(), krnl_args.begin(), krnl_args.end());
    _task_queue.push_back(std::move(task));
  }

  void synchronize(void *stream) {
    if (_task_queue.size()) {
      const int max_allowed_threads = _thread_pool->get_workers().size();
      std::vector<std::future<void>> results;
      for (auto &it: _task_queue) {
        auto func = ((void(*)(int, void* const*))it[0]);
        auto num_threads = (int)(long)it[1];
        auto args = it.data() + 2;

        for (int i = 0; i < max_allowed_threads; ++i)
          results.emplace_back(_thread_pool->enqueue([=]() -> void {
            for (int j = i; j < num_threads; j += max_allowed_threads)
              func(j, args);
          }));
      }
      for (auto &it: results)
        it.wait();
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
