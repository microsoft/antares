// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>
#include <vector>
#include <string>
#include <map>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime_api.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime_api.h>

#define CUmodule hipModule_t
#define CUfunction hipFunction_t

#define cuModuleLoad hipModuleLoad
#define cuModuleUnload hipModuleUnload
#define cuModuleGetFunction hipModuleGetFunction
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR hipDeviceAttributeComputeCapabilityMajor
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR hipDeviceAttributeComputeCapabilityMinor

#define cuLaunchKernel(f, bx, by, bz, tx, ty, tz, shm, stream, args, extra) \
	        hipModuleLaunchKernel(f, bx, by, bz, tx, ty, tz, shm, stream, args, extra)

#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaStream_t hipStream_t
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventDestroy hipEventDestroy
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventDisableTiming 0

#endif

static std::map<std::string, std::pair<CUmodule, CUfunction>> module_manager;

std::vector<torch::Tensor> custom_op_forward(std::vector<torch::Tensor> inputs,
                                             const std::string& source,
                                             const std::string& source_path,
                                             const std::string& hash,
                                             const std::vector<std::string>& meta_inputs,
                                             const std::vector<std::string>& meta_outputs)
{
  CUmodule hmod = nullptr;
  CUfunction hfunc = nullptr;

  auto it = module_manager.find(hash);
  if (it == module_manager.end())
  {
    std::string kernel_src_path = source_path, kernel_path = source_path + ".out";
    FILE *fp = fopen(kernel_src_path.c_str(), "wb");
    CHECK_EQ(source.size(), fwrite(source.c_str(), 1, source.size(), fp));
    fclose(fp);

    int major, minor;
#ifndef __HIP_PLATFORM_HCC__
    CHECK_EQ(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0), 0);
    CHECK_EQ(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0), 0);
    std::string arch = std::to_string(major * 10 + minor);
    std::string compile_cmd = "/usr/local/cuda/bin/nvcc " + kernel_src_path + " -gencode arch=compute_" + arch + ",code=sm_" + arch + " --fatbin -O2 -o " + kernel_path;
#else
    static hipDeviceProp_t prop;
    if (!prop.gcnArch)
      CHECK_EQ(hipGetDeviceProperties(&prop, 0), 0);
    major = prop.gcnArch / 100, minor = prop.gcnArch % 100;

    std::string arch = std::to_string(major * 100 + minor);
    std::string compile_cmd = "/opt/rocm/bin/hipcc " + kernel_src_path + " --amdgpu-target=gfx" + arch + " --genco -Wno-ignored-attributes -O2 -o " + kernel_path;
#endif
    LOG(INFO) << "MainOpKernel is compiling dynamtic kernel (arch=" << arch << "): " << kernel_path;
    CHECK_EQ(system(compile_cmd.c_str()), 0);

    CHECK_EQ(cuModuleLoad(&hmod, kernel_path.c_str()), 0);
    CHECK_EQ(cuModuleGetFunction(&hfunc, hmod, "template_op_kernel0"), 0);
    module_manager[hash] = {hmod, hfunc};
  }
  else
    hmod = it->second.first, hfunc = it->second.second;

  int bx, by, bz, tx, ty, tz;
  int pos, next;
  pos = source.find("// [thread_extent] blockIdx.x"), next = source.find("= ", pos + 1), bx = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;
  pos = source.find("// [thread_extent] blockIdx.y"), next = source.find("= ", pos + 1), by = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;
  pos = source.find("// [thread_extent] blockIdx.z"), next = source.find("= ", pos + 1), bz = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;
  pos = source.find("// [thread_extent] threadIdx.x"), next = source.find("= ", pos + 1), tx = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;
  pos = source.find("// [thread_extent] threadIdx.y"), next = source.find("= ", pos + 1), ty = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;
  pos = source.find("// [thread_extent] threadIdx.z"), next = source.find("= ", pos + 1), tz = (pos >= 0 && next >= 0) ? std::atoi(source.c_str() + next + 2) : 1;

  std::vector<void*> args, p_args;
  pos = source.find(") {\n"), next = source.rfind('(', pos) + 1;
  CHECK_EQ(true, (pos > 0 && next > 0));
  auto code_args = source.substr(next, pos - next) + ",";
  args.resize(meta_inputs.size() + meta_outputs.size()), p_args.resize(args.size());
  for (int i = 0; i < args.size(); ++i)
    p_args[i] = &args[i];

  std::vector<std::vector<int64_t>> output_shapes;
  output_shapes.clear();
  for (int y = 0; y < meta_outputs.size(); ++y) {
    auto meta_shape = meta_outputs[y].substr(0, meta_outputs[y].find('/')) + "-";
    std::vector<int64_t> shape_builder;
    for (int i = 0, j = 1; j < meta_shape.size(); ++j) {
      if (meta_shape[j] == '-')
        shape_builder.push_back(std::atoi(meta_shape.c_str() + i)), i = j + 1;
    }
    output_shapes.push_back(std::move(shape_builder));
  }

  std::vector<torch::Tensor> outputs;
  outputs.resize(meta_outputs.size());
  auto options =
    torch::TensorOptions()
      .dtype(inputs[0].dtype())
      .device(inputs[0].device().type(), inputs[0].device().index())
      .layout(torch::kStrided)
      .requires_grad(true);

  for (int i = 0; i < outputs.size(); ++i) {
    outputs[i] = torch::zeros(output_shapes[i], options);
  }

  for (int i = 0; i < inputs.size(); ++i)
  {
    args[i] = (void*)inputs[i].data_ptr();
  }
  for (int i = 0; i < meta_outputs.size(); ++i)
  {
    args[meta_inputs.size() + i] = (void*)outputs[i].data_ptr();
  }

  CHECK_EQ(cuLaunchKernel(hfunc, bx, by, bz, tx, ty, tz, 0, 0, p_args.data(), NULL), 0);

  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_op_forward, "custom forward (GPU)");
}
