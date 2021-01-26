// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

#include <unistd.h>
#include <CL/sycl.hpp>

#include <my_kernel.cc>

std::string get_between(const std::string &str, const std::string &begin, const std::string &end, int start_idx = 0, const std::string &def_ret = "") {
    if (start_idx < 0)
        return def_ret;
    int at = str.find(begin);
    if (at < 0)
        return def_ret;
    at += begin.size();
    int next = str.find(end, at);
    if (next < at)
        return def_ret;
    return str.substr(at, next - at);
}

std::vector<std::string> ssplit(const std::string &str, const std::string &sub) {
    std::vector<std::string> ret;
    int it = 0, next;
    while (next = str.find(sub, it), next >= 0) {
        ret.push_back(str.substr(it, next - it));
        it = next + sub.size();
    }
    ret.push_back(str.substr(it));
    return std::move(ret);
}

struct tensor_property {
    std::string name, dtype;
    std::vector<size_t> shape;

    size_t element_size() const {
        return std::accumulate(shape.begin(), shape.end(), (size_t)1L, std::multiplies<size_t>());
    }

    int type_size() const {
        for (int i = dtype.size() - 1; i >= 0; --i) {
            if (!isdigit(dtype[i])) {
                int bits = std::atoi(dtype.substr(i + 1).c_str());
                assert((bits & 7) == 0);
                return bits >> 3;
            }
        }
        throw std::runtime_error(("Unrecognized type size for `" + dtype + "`").c_str());
    }
};

std::vector<tensor_property> parse_properties(const std::string &encoded_inputs) {
    if (encoded_inputs.size() == 0)
      return {};
    std::vector<tensor_property> ret;
    for (auto it: ssplit(encoded_inputs, ",")) {
      auto props = ssplit(it, "/");
      tensor_property tp;
      auto sshape = ssplit(props[0], "-");
      for (auto d: sshape)
        tp.shape.push_back(std::atol(d.c_str()));
      tp.dtype = props[1];
      tp.name = props[2];
      ret.push_back(tp);
    }
    return std::move(ret);
}

void *timeout_monitor(void *arg) {
    sleep(30);
    fprintf(stderr, "[FATAL] Time limit exceeded for this evaluation.\n");
    _exit(1);
}

int main(int argc, char** argv)
{
    pthread_t p_timeout_monitor;
    pthread_create(&p_timeout_monitor, NULL, timeout_monitor, NULL);
    pthread_detach(p_timeout_monitor);

    using namespace sycl;

    std::ifstream t("my_kernel.cc");
    std::string source((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();

    auto encoded_params = get_between(source, "///", "\n");
    auto params = ssplit(encoded_params, ":");
    auto inputs = parse_properties(params[0]), outputs = parse_properties(params[1]);

    std::vector<void*> h_args;
    for (int i = 0; i < inputs.size(); ++i) {
      auto &it = inputs[i];
      h_args.push_back(new char[it.element_size() * it.type_size()]);
      void* ptrs = h_args.back();

      size_t size = it.element_size();
      if (it.dtype == "int32") {
        for (size_t x = 0; x < size; ++x)
          ((int*)(ptrs))[x] = (x + i + 1) % 71;
      } else if (it.dtype == "float32") {
        for (size_t x = 0; x < size; ++x)
          ((float*)(ptrs))[x] = (x + i + 1) % 71;
      } else {
        size_t byte_size = size * it.type_size();
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          ((int*)(ptrs))[x] = (x + i + 1) % 71;
        for (size_t x = byte_size - byte_size % sizeof(int); x < byte_size; x++)
          ((char*)(ptrs))[x] = 1;
      }
    }
    for (auto it: outputs) {
      h_args.push_back(new char[it.element_size() * it.type_size()]);
      void* ptrs = h_args.back();

      memset(ptrs, 0, it.element_size() * it.type_size());
    }

    try {
      queue q(default_selector{});
      printf("\nDevice Name: %s\n\n", q.get_device().get_info<info::device::name>().c_str());

      auto start = std::chrono::high_resolution_clock::now();
      compute_kernel_vargs(&q, h_args.data());
      q.wait();
      auto end = std::chrono::high_resolution_clock::now();
      double tpr = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      for (int c = 0; c < outputs.size(); ++c) {
        size_t byte_size = outputs[c].element_size() * outputs[c].type_size();
        double digest = 0.0;
        if (outputs[c].dtype == "int32") {
          for (size_t i = 0; i < byte_size / sizeof(int); ++i)
            digest += (i + 1) % 83 * ((int*)h_args[inputs.size() + c])[i];
        } else {
          for (size_t i = 0; i < byte_size / sizeof(float); ++i)
            digest += (i + 1) % 83 * ((float*)h_args[inputs.size() + c])[i];
          for (size_t i = byte_size - byte_size % sizeof(int); i < byte_size; i++)
            digest += ((char*)h_args[inputs.size() + c])[i];
        }
        printf("- K/%d: %.10e\n", c, digest);
      }

      int recommended_runs = std::min(std::max(1.0 / tpr, 1.0), 1.0e4);
      start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < recommended_runs; ++i) {
        compute_kernel_vargs(&q, h_args.data());
        q.wait();
      }
      end = std::chrono::high_resolution_clock::now();

      tpr = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / recommended_runs;
      printf("- TPR: %g\n", tpr);

    } catch (sycl::exception const &e) {
      std::terminate();
    }
    return 0;
}
