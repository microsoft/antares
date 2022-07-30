// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-ocl_android): [aarch64-linux-android-clang++] -ldl -O2

#include <dlfcn.h>

#define eprintf(...) fprintf (stderr, __VA_ARGS__)

typedef void *cl_device_id, *cl_context, *cl_command_queue, *cl_mem, *cl_platform_id, *cl_program, *cl_kernel;
typedef int cl_int;
typedef unsigned int cl_uint;

namespace cl_loader {
struct android_initializor {
  void *lib;
  android_initializor() {
    lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_LOCAL);
    CHECK_OK(lib != nullptr);
  }
};

android_initializor _;
} // namespace cl_loader

#define LOAD_SYMB(type, func)    auto func = (type(*)(...))(dlsym(cl_loader::_.lib, #func));

#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
#define CL_DEVICE_NAME                              0x102B
#define CL_DEVICE_MAX_WORK_GROUP_SIZE               0x1004
#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_PROGRAM_BUILD_LOG                        0x1183
#define CL_FALSE                                    0

namespace ab {
  static cl_context context;
  static cl_command_queue cmdqueue;
  static cl_device_id device_id;
  static cl_int stat;
  static size_t max_work_group_size;

  void init(int dev) {
    LOAD_SYMB(int, clGetPlatformIDs);
    LOAD_SYMB(int, clGetDeviceIDs);
    LOAD_SYMB(int, clGetDeviceInfo);
    LOAD_SYMB(void*, clCreateContext);
    LOAD_SYMB(void*, clCreateCommandQueue);

    cl_uint num_dev;
    CHECK_OK(0 == clGetPlatformIDs(0, 0, &num_dev));

    std::vector<cl_platform_id> platforms(num_dev);
    CHECK_OK(0 == clGetPlatformIDs(num_dev, &platforms[0], &num_dev));
    CHECK_OK(num_dev > 0);

    CHECK_OK(0 == clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, 0, &num_dev));
    std::vector<cl_device_id> devices(num_dev);
    CHECK_OK(dev < num_dev);
    CHECK_OK(0 == clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_dev, &devices[0], &num_dev));

    device_id = devices[dev];
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &stat), CHECK_OK(stat == 0);
    cmdqueue = clCreateCommandQueue(context, device_id, 0, &stat), CHECK_OK(stat == 0);

    std::vector<char> dev_name(1024);
    CHECK_OK(0 == clGetDeviceInfo(device_id, CL_DEVICE_NAME, dev_name.size(), dev_name.data(), NULL));
    CHECK_OK(0 == clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL));
    fprintf(stderr, "    (OCL_INFO: OCL Device Name = %s [max_work_groups: %zd])\n", dev_name.data(), max_work_group_size);
  }

  void finalize() {
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    LOAD_SYMB(void*, clCreateBuffer);

    cl_mem dptr = clCreateBuffer(context, CL_MEM_READ_WRITE, byteSize, NULL, &stat); CHECK_OK(stat == 0);
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    LOAD_SYMB(int, clReleaseMemObject);

    CHECK_OK(0 == clReleaseMemObject((cl_mem)dptr));
  }

  std::string moduleCompile(const std::string &source) {
    return source;
  }

  void* moduleLoad(const std::string &binary) {
    LOAD_SYMB(void*, clCreateProgramWithSource);
    LOAD_SYMB(int, clBuildProgram);
    LOAD_SYMB(int, clGetProgramBuildInfo);

    const char *source_data = binary.data();
    size_t source_size = binary.size();
    cl_program program = clCreateProgramWithSource(context, 1, &source_data, &source_size, &stat);
    CHECK_OK(stat == 0);
    if (0 != clBuildProgram(program, 1, &device_id, NULL, NULL, NULL)) {
      size_t log_size = 0;
      CHECK_OK(0 == clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
      std::vector<char> log_data(log_size + 1);
      CHECK_OK(0 == clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, (char*)log_data.data(), nullptr));
      fprintf(stderr, "[Build Error] %s\n", log_data.data());
      CHECK_OK(0);
    }
    return program;
  }

  std::vector<void*> moduleGetFunction(const void *hModule, const std::string &fname, const std::unordered_map<std::string, int> &threads) {
    LOAD_SYMB(void*, clCreateKernel);

    cl_kernel kernel = clCreateKernel((cl_program)hModule, fname.c_str(), &stat);
    CHECK_OK(stat == 0);

    auto query = [&](const std::string &axis, size_t defval = 1) -> size_t {
      auto it = threads.find(axis);
      if (it == threads.end())
        return defval;
      return it->second;
    };
    size_t lx = query("get_local_id(0)"), ly = query("get_local_id(1)"), lz = query("get_local_id(2)");
    size_t gx = query("get_group_id(0)"), gy = query("get_group_id(1)"), gz = query("get_group_id(2)");

    CHECK_OK(lx * ly * lz <= max_work_group_size);
    return { kernel, (void*)(gx * lx), (void*)(gy * ly), (void*)(gz * lz), (void*)lx, (void*)ly, (void*)lz };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args, void *stream) {
    LOAD_SYMB(int, clSetKernelArg);
    LOAD_SYMB(int, clEnqueueNDRangeKernel);

    auto kernel = (cl_kernel)hFunction[0];
    for (int i = 0; i < krnl_args.size(); ++i)
      CHECK_OK(0 == clSetKernelArg(kernel, i, sizeof(cl_mem), (void*)&krnl_args[i]));
    CHECK_OK(0 == clEnqueueNDRangeKernel(cmdqueue, kernel, 3, nullptr, (size_t*)(hFunction.data() + 1), (size_t*)(hFunction.data() + 4), 0, nullptr, nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    LOAD_SYMB(int, clEnqueueWriteBuffer);

    CHECK_OK(0 == clEnqueueWriteBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_write */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    LOAD_SYMB(int, clEnqueueReadBuffer);

    CHECK_OK(0 == clEnqueueReadBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_read */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void synchronize(void *stream) {
    LOAD_SYMB(int, clFinish);

    CHECK_OK(0 == clFinish(cmdqueue));
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
    delete h1; delete h2;
    return std::max(et, 1e-9);
  }
}

