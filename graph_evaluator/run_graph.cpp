// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifdef _MSC_VER
#define NOMINMAX
#endif
#include "npy.hpp"
#include "execute_module.hpp"

static void *memory_alloc(size_t length)
{
#if defined(HLSL_LIBRARY_PATH)
    void *data_ptr = (void*)GlobalAlloc(0, length);
#else
    void *data_ptr = (void*)memalign(256, length);
#endif
    return data_ptr;
}

static void memory_free(void *data_ptr)
{
    if (!data_ptr)
        return;
#if defined(HLSL_LIBRARY_PATH)
    GlobalFree(data_ptr);
#else
    free(data_ptr);
#endif
}

int main(int argc, char** argv)
{
    float expected_timeout = -1;
    use_progress = 0;
    int dev = 0, compile = 0;
    std::string value_absdir;

    for (int i = 2; i < argc; ++i) {
      if (!strcmp(argv[i], "--progress"))
        use_progress = 1;
      else if (!strcmp(argv[i], "--debug"))
        debug_output = std::atoi(argv[++i]);
      else if (!strcmp(argv[i], "--compile"))
        compile = 1;
      else if (!strcmp(argv[i], "--timeout"))
        expected_timeout = std::atof(argv[++i]);
      else if (!strcmp(argv[i], "--value_absdir")) {
        value_absdir = argv[++i];
        if (value_absdir.size() > 0 && value_absdir.back() != '/' && value_absdir.back() != '\\')
          value_absdir += "/";
      } else if (!strcmp(argv[i], "--dev"))
        dev = std::atoi(argv[++i]);
      else {
        fprintf(stderr, "[Error] Unrecognized keyword: %s\n", argv[i]);
        return 1;
      }
    }
    typedef long long llong;

#if !defined(_WIN64) || defined(__MINGW64__)
    pthread_t p_timeout_monitor;
    pthread_create(&p_timeout_monitor, NULL, [](void *arg) -> void* {
      float &expected_timeout = *(float*)arg;
      if (expected_timeout <= 0)
        return nullptr;
      int timeout_sec = 12 + expected_timeout;
      sleep(timeout_sec);
      if (!use_progress)
        fprintf(stderr, "[FATAL] Time limit exceeded (>= %d sec) for this evaluation.\n", timeout_sec);
      exit(1);
      return nullptr;
    }, &expected_timeout);
    pthread_detach(p_timeout_monitor);
#endif

    const char *module_path = argc > 1 ? argv[1] : "./my_kernel.cc";

    auto src = ExecutionModule::load_source(std::string("file://") + module_path);

    if (compile) {
      ab::init(0); // Useful to initialize libraries.
      auto binary = ab::moduleCompile(src);
      printf("\n- HEX: @");
      for (int i = 0; i < binary.size(); ++i)
        printf("%02X", ((unsigned char)binary[i]));
      printf("@\n"), fflush(stdout);
      ab::finalize();
      return 0;
    }

    ExecutionModule gm(src, dev);
    std::vector<void*> global_args;
    for (int i = 0; i < gm.global_inputs.size(); ++i) {
      auto &it = gm.global_inputs[i];
      void *dptr = allocate_tensor(it);
      global_args.push_back(dptr);

      void *data_ptr = memory_alloc(it.mem_size());
      size_t size = it.element_size();

      auto numpy_load = [&](const std::string &path, size_t byte_size) -> bool {
        if (value_absdir.size() == 0)
          return false;
        std::ifstream stream(path, std::ifstream::in | std::ifstream::binary);
        if (!stream)
         return false;
        // CHECK_OK(header.dtype.kind == 'f' && header.dtype.itemsize == 8);

        bool fortran_order;
        using namespace npy;
        std::string header_s = read_header(stream);
        header_t header = parse_header(header_s); // header.dtype.str().c_str()
        std::vector<unsigned long> shape = header.shape;
        ssize_t size = static_cast<ssize_t>(comp_size(shape));

        size_t fbytes = size * header.dtype.itemsize;
        size_t read_size = std::min(byte_size, fbytes);
        stream.read(reinterpret_cast<char*>(data_ptr), read_size);
        for (int i = fbytes; i < byte_size; ++i)
          ((char*)data_ptr)[i] = ((char*)data_ptr)[i - fbytes];
        return true;
      };

      if (numpy_load(value_absdir + it.name + ".npy", it.mem_size())) {
        // pass;
      } else if (it.dtype == "int32") {
        for (size_t x = 0; x < size; ++x)
          ((int*)data_ptr)[x] = 0;
      } else if (it.dtype == "int16") {
        for (size_t x = 0; x < size; ++x)
          ((short*)data_ptr)[x] = 0;
      } else if (it.dtype == "int64") {
        for (size_t x = 0; x < size; ++x)
          ((llong*)data_ptr)[x] = 0;
      } else if (it.dtype == "float16") {
        for (size_t x = 0; x < size; ++x)
          ((unsigned short*)data_ptr)[x] = fp32_to_fp16((x + i + 1) % 71);
      } else if (it.dtype == "float32") {
        for (size_t x = 0; x < size; ++x)
          ((float*)data_ptr)[x] = (x + i + 1) % 71;
      } else if (it.dtype == "float64") {
        for (size_t x = 0; x < size; ++x)
          ((double*)data_ptr)[x] = (x + i + 1) % 71;
      } else {
        size_t byte_size = size * it.type_size();
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          ((int*)data_ptr)[x] = (x + i + 1) % 71;
        for (size_t x = byte_size - byte_size % sizeof(int); x < byte_size; x++)
          ((char*)data_ptr)[x] = 1;
      }
      ab::memcpyHtoD(dptr, data_ptr, it.mem_size(), nullptr);
      ab::synchronize(nullptr);
      memory_free(data_ptr);
    }
    for (auto &it: gm.global_outputs) {
      void *dptr = allocate_tensor(it);
      global_args.push_back(dptr);

      void *data_ptr = memory_alloc(it.mem_size());
      memset(data_ptr, 0, it.mem_size());
      ab::memcpyHtoD(dptr, data_ptr, it.mem_size(), nullptr);
      ab::synchronize(nullptr);
      memory_free(data_ptr);
    }
    int expanded_args = 0;
    if (gm.vamap.size() > 0) {
      char *p = strtok((char*)gm.vamap.data(), ",");
      while (p) {
        p = strchr(p, ':') + 1;
        global_args.push_back((void*)(size_t)std::atoi(p));
        p = strtok(nullptr, ",");
        ++expanded_args;
      }
    }

    gm.compute(global_args.data(), expanded_args);

    FILE *fp = fopen("stdout.log", "wb");
    CHECK_OK(fp != nullptr);

    for (int i = 0; i < gm.global_outputs.size(); ++i) {
      auto &it = gm.global_outputs[i];
      void *dptr = global_args[gm.global_inputs.size() + i];

      void *data_ptr = memory_alloc(it.mem_size());
      ab::memcpyDtoH(data_ptr, dptr, it.mem_size(), nullptr);
      ab::synchronize(nullptr);

      size_t byte_size = it.mem_size();
      double digest = 0.0;
      if (it.dtype == "int32") {
        for (size_t x = 0; x < byte_size / sizeof(int); ++x)
          digest += (x + 1) % 83 * ((int*)data_ptr)[x];
      } else if (it.dtype == "int64") {
        for (size_t x = 0; x < byte_size / sizeof(llong); ++x)
          digest += (x + 1) % 83 * ((llong*)data_ptr)[x];
      } else if (it.dtype == "float16") {
        for (size_t x = 0; x < byte_size / sizeof(unsigned short); ++x)
          digest += (x + 1) % 83 * fp16_to_fp32(((unsigned short*)data_ptr)[x]);
      } else if (it.dtype == "float32") {
        for (size_t x = 0; x < byte_size / sizeof(float); ++x)
          digest += (x + 1) % 83 * ((float*)data_ptr)[x];
      } else if (it.dtype == "float64") {
        for (size_t x = 0; x < byte_size / sizeof(double); ++x)
          digest += (x + 1) % 83 * ((double*)data_ptr)[x];
      } else {
        for (size_t x = 0; x < byte_size; ++x)
          digest += (x + 1) % 83 * ((unsigned char*)data_ptr)[x];
      }
      printf("\n- K/%d: %.10e\n", i, digest), fflush(stdout);
      fprintf(fp, "\n- K/%d: %.10e\n", i, digest), fflush(fp);
      memory_free(data_ptr);
    }

    do {
      auto x = ab::recordTime(nullptr);
      gm.compute(global_args.data(), expanded_args);
      auto y = ab::recordTime(nullptr);
      ab::synchronize(nullptr);

      double tpr = ab::convertToElapsedTime(x, y);
      if ((expected_timeout > 0 && tpr > expected_timeout) || tpr > 2) {
        printf("\n- TPR: %g\n", tpr), fflush(stdout);
        fprintf(fp, "\n- TPR: %g\n", tpr), fflush(fp);
        break;
      }

      int num_runs = (int)std::max(1LU, std::min(10000LU, (unsigned long)(1.0 / tpr)));
      tpr = 0.0f;
      x = ab::recordTime(nullptr);
      for (int i = 0; i < num_runs; ++i)
        gm.compute(global_args.data(), expanded_args);
      y = ab::recordTime(nullptr);
      tpr = ab::convertToElapsedTime(x, y) / num_runs;
      printf("\n- TPR: %g\n", tpr), fflush(stdout);
      fprintf(fp, "\n- TPR: %g\n", tpr), fflush(fp);
    } while (0);

    fclose(fp);
    ab::finalize();
    return 0;
}
