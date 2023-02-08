// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//; eval_flags(c-ocl_amdgpu): -I/opt/rocm/opencl/include -L/opt/rocm/opencl/lib -lOpenCL -DCL_TARGET_OPENCL_VERSION=120
//; eval_flags(c-ocl_nvidia): -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lOpenCL
//; eval_flags(c-ocl_intel): -I/opt/intel/oneapi/compiler/latest/linux/include -L/opt/intel/oneapi/compiler/latest/linux/lib -lOpenCL -Wno-deprecated-declarations

#include <CL/cl.h>

namespace ab {
  static cl_context context;
  static cl_command_queue cmdqueue;
  static cl_device_id device_id;
  static cl_int stat;
  static size_t max_work_group_size;

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

    std::vector<char> dev_name(1024);
    CHECK_OK(0 == clGetDeviceInfo(device_id, CL_DEVICE_NAME, dev_name.size(), dev_name.data(), NULL));
    CHECK_OK(0 == clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL));
    fprintf(stderr, "    (OCL_INFO: OCL Device Name = %s [max_work_groups: %zd])\n", dev_name.data(), max_work_group_size);
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

  std::string moduleCompile(const std::string &source) {
    return source;
  }

  void* moduleLoad(const std::string &binary) {
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
    std::vector<void*> fdata = { kernel, (void*)(gx * lx), (void*)(gy * ly), (void*)(gz * lz), (void*)lx, (void*)ly, (void*)lz };

    void *item = (void*)query("$", 0);
    if (item) {
      fdata.push_back(item);

      for (int i = 0; ; ++i) {
        void *item = (void*)query("$" + std::to_string(i), 0);
        if (!item)
          break;
        fdata.push_back(item);
      }
    }

    return fdata;
  }

  void launchKernel(std::vector<void*> &hFunc, const std::vector<void*> &krnl_args, void *stream) {
    if (hFunc.size() > 7) {
      long attrs = 1;
      for (int i = 8; i < hFunc.size(); ++i) {
        long val = (long)hFunc[i];
        if (val < 0) continue;

        auto ptr = (size_t*)&krnl_args[i - 8 + (long)hFunc[7]];
        attrs *= (*ptr + val - 1) / val;
      }
      hFunc[1] = (void*)(size_t(hFunc[4]) * attrs);
      if (!hFunc[1]) return;
    }

    auto kernel = (cl_kernel)hFunc[0];
    for (int i = 0; i < krnl_args.size(); ++i) {
      if (hFunc.size() > 7 && i >= (long)hFunc[7])
        CHECK_OK(0 == clSetKernelArg(kernel, i, sizeof(cl_uint), (void*)&krnl_args[i]) || i + 1 == krnl_args.size());
      else
        CHECK_OK(0 == clSetKernelArg(kernel, i, sizeof(cl_mem), (void*)&krnl_args[i]));
    }
    CHECK_OK(0 == clEnqueueNDRangeKernel(cmdqueue, kernel, 3, nullptr, (size_t*)(hFunc.data() + 1), (size_t*)(hFunc.data() + 4), 0, nullptr, nullptr));
  }

  void memcpyHtoD(void *dptr, void *hptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == clEnqueueWriteBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_write */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void memcpyDtoH(void *hptr, void *dptr, size_t byteSize, void *stream) {
    CHECK_OK(0 == clEnqueueReadBuffer(cmdqueue, (cl_mem)dptr, CL_FALSE /* blocking_read */, 0, byteSize, hptr, 0, NULL, NULL));
  }

  void synchronize(void *stream) {
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
    delete h1, h2;
    return std::max(et, 1e-9);
  }
}

