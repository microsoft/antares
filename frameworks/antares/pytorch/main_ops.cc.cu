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

#define cuLaunchKernel(f, bx, by, bz, tx, ty, tz, shm, stream, args, extra) \
	        hipModuleLaunchKernel(f, bx, by, bz, tx, ty, tz, shm, stream, args, extra)

#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaStream_t hipStream_t
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventDestroy hipEventDestroy
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventDisableTiming 0

#endif

struct ModuleItem
{
  ModuleItem() {}
  ModuleItem(CUmodule m, CUfunction f)
    : hmod(m),
      hfunc(f) {}
  CUmodule hmod = nullptr;
  CUfunction hfunc = nullptr;
};

static std::map<std::string, ModuleItem> module_manager;

std::vector<torch::Tensor> custom_op_forward(std::vector<torch::Tensor> inputs,
                                             const std::string& source,
                                             const std::string& kernel_path,
                                             const std::string& hash,
                                             const std::vector<std::string>& meta_inputs,
                                             const std::vector<std::string>& meta_outputs)
{
  LOG(INFO) << "MainOpKernel is compiling the dynamtic kernel..";

  CUmodule hmod = nullptr;
  CUfunction hfunc = nullptr;

  if (module_manager.count(hash))
  {
    hmod = module_manager[hash].hmod;
    hfunc = module_manager[hash].hfunc;
  }
  else
  {
    CHECK_EQ(cuModuleLoad(&hmod, kernel_path.c_str()), 0);
    CHECK_EQ(cuModuleGetFunction(&hfunc, hmod, "template_op_kernel0"), 0);
    module_manager[hash] = ModuleItem(hmod, hfunc);
  }

  int bx, by, bz, tx, ty, tz;
  int i, pos, next;
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
  for (i = pos = 0; next = code_args.find(',', pos), next >= 0; pos = next + 1, ++i) {
    int at = code_args.rfind(' ', next) + 1;
    auto arg_name = code_args.substr(at, next - at);
    CHECK_NE(arg_name, "");
    if (arg_name[0] == 'i')
      p_args[i] = &args[std::atoi(arg_name.c_str() + 5)];
    else
      p_args[i] = &args[meta_inputs.size() + std::atoi(arg_name.c_str() + 6)];
  }

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
