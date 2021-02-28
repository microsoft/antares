// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-ocl_amdgpu): -I/opt/rocm/opencl/include -L/opt/rocm/opencl/lib -lOpenCL -DCL_TARGET_OPENCL_VERSION=120

#include <CL/cl.h>

namespace ab {
  static cl_context context;
  static cl_command_queue cmdqueue;
  static cl_device_id device_id;
  static cl_int stat;

  void init(int dev) {
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
  }

  void finalize() {
  }

  void* alloc(size_t byteSize, const std::vector<size_t> &shape, const std::string &dtype, const std::string &name) {
    cl_mem dptr = clCreateBuffer(context, CL_MEM_READ_WRITE, byteSize, NULL, &stat); CHECK_OK(stat == 0);
    return dptr;
  }

  void release(void *dptr, size_t byteSize) {
    CHECK_OK(0 == clReleaseMemObject((cl_mem)dptr));
  }

  void* moduleLoad(const std::string &source) {
    const char *source_data = source.data();
    size_t source_size = source.size();
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
    return { kernel, (void*)(gx * lx), (void*)(gy * ly), (void*)(gz * lz), (void*)lx, (void*)ly, (void*)lz };
  }

  void launchKernel(const std::vector<void*> &hFunction, const std::vector<void*> &krnl_args) {
    auto kernel = (cl_kernel)hFunction[0];
    for (int i = 0; i < krnl_args.size(); ++i)
      CHECK_OK(0 == clSetKernelArg(kernel, i, sizeof(cl_mem), (void*)&krnl_args[i]));
    CHECK_OK(0 == clEnqueueNDRangeKernel(cmdqueue, kernel, 3, nullptr, (size_t*)(hFunction.data() + 1), (size_t*)(hFunction.data() + 4), 0, nullptr, nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize) {
    CHECK_OK(0 == clEnqueueWriteBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_write */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize) {
    CHECK_OK(0 == clEnqueueReadBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_read */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void synchronize() {
    CHECK_OK(0 == clFinish(cmdqueue));
  }

  void* recordTime() {
    ab::synchronize();

    auto pt = new std::chrono::high_resolution_clock::time_point;
    *pt = std::chrono::high_resolution_clock::now();
    return pt;
  }

  double convertToElapsedTime(void *hStart, void *hStop) {
    ab::synchronize();

    auto h1 = (std::chrono::high_resolution_clock::time_point*)hStart;
    auto h2 = (std::chrono::high_resolution_clock::time_point*)hStop;

    double et = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(*h2 - *h1).count();
    delete h1, h2;
    return std::max(et, 1e-9);
  }
}

