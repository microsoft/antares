// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_runtime_api.h>
#include <nccl.h>
#else
#include <hip/hip_runtime_api.h>
#include <rccl.h>

#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaStream_t hipStream_t
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventDestroy hipEventDestroy
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventDisableTiming 0

#endif

#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>

#include <mpi.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>


#if !defined(__linux__)
#error "Only Linux platform is supported at the moment (with CUDA)."
#endif

namespace tensorflow {
namespace {

using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


class Nccl2Handle {
 public:
  Nccl2Handle() {
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    LOG(INFO) << "Nccl2Handle Initialize: device-rank = " << mpi_rank;
    ncclUniqueId id;
    if (mpi_rank == 0)
      CHECK_EQ(ncclSuccess, ncclGetUniqueId(&id));
    CHECK_EQ(MPI_SUCCESS, MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_EQ(ncclSuccess, ncclGroupStart());
    CHECK_EQ(ncclSuccess, ncclCommInitRank(&comm, mpi_size, id, mpi_rank));
    CHECK_EQ(ncclSuccess, ncclGroupEnd());
   }

  ncclComm_t getHandle() const {
    return comm;
  }

  ~Nccl2Handle() {
    LOG(INFO) << "Nccl2Handle Destroy inter-session communication: device-rank = " << mpi_rank;
    CHECK_EQ(ncclSuccess, ncclCommDestroy(comm));
  }

  int mpi_size, mpi_rank;
  ncclComm_t comm;
};


static shared_ptr<Nccl2Handle> __ncclComm;
static pthread_mutex_t __g_lock = PTHREAD_MUTEX_INITIALIZER;

static shared_ptr<Nccl2Handle> initializeNccl2() {
  pthread_mutex_lock(&__g_lock);
  if (__ncclComm == nullptr)
    __ncclComm = make_shared<Nccl2Handle>();
  pthread_mutex_unlock(&__g_lock);
  return __ncclComm;
}

static void finalizeNccl2() {
  pthread_mutex_lock(&__g_lock);
  if (__ncclComm.use_count() <= 1)
    __ncclComm = nullptr;
  pthread_mutex_unlock(&__g_lock);
}

static void loadTypeConfig(OpKernelConstruction* c, ncclDataType_t &data_type, ncclRedOp_t &reduce_type) {
  std::string _data_type, _reduce_type;
  OP_REQUIRES_OK(c, c->GetAttr("data_type", &_data_type));
  OP_REQUIRES_OK(c, c->GetAttr("reduce_type", &_reduce_type));

  if (_data_type == "float32")
    data_type = ncclFloat32;
  else if (_data_type == "int32")
    data_type = ncclInt32;
  else
    throw std::runtime_error(("Unhandled data_type for communication: " + _data_type).c_str());

  if (_reduce_type == "")
    reduce_type = (ncclRedOp_t)-1;
  else if (_reduce_type == "+")
    reduce_type = ncclSum;
  else if (_reduce_type == ">")
    reduce_type = ncclMax;
  else if (_reduce_type == "<")
    reduce_type = ncclMin;
  else
    throw std::runtime_error(("Unhandled reduce_type for communication: " + _reduce_type).c_str());
}

/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class Nccl2AllreduceOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2AllreduceOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    loadTypeConfig(c, data_type, reduce_type);
  }

  ~Nccl2AllreduceOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, c->input(i).shape(), &output), done);
      CHECK_EQ(ncclSuccess, ncclAllReduce((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), data_type, reduce_type, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  ncclDataType_t data_type;
  ncclRedOp_t reduce_type;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2AllreduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Allreduce").Device(DEVICE_GPU), Nccl2AllreduceOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Allreduce")
    .Input("tensor: N * T")
    .Output("sum: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("data_type: string")
    .Attr("reduce_type: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i)
        c->set_output(i, c->input(i));
      return Status::OK();
    });


/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class Nccl2ReducescatterOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2ReducescatterOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    loadTypeConfig(c, data_type, reduce_type);
    OP_REQUIRES_OK(c, c->GetAttr("node_size", &node_size));
  }

  ~Nccl2ReducescatterOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      auto result_shape = c->input(i).shape();
      CHECK_EQ(0, result_shape.dim_size(0) % node_size);
      result_shape.set_dim(0, result_shape.dim_size(0) / node_size);

      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, result_shape, &output), done);
      CHECK_EQ(ncclSuccess, ncclReduceScatter((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements() / node_size, data_type, reduce_type, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  int node_size;
  ncclDataType_t data_type;
  ncclRedOp_t reduce_type;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2ReducescatterOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Reducescatter").Device(DEVICE_GPU), Nccl2ReducescatterOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Reducescatter")
    .Input("tensor: N * T")
    .Output("sum: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("node_size: int")
    .Attr("data_type: string")
    .Attr("reduce_type: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i)
        c->set_output(i, c->UnknownShape());
      return Status::OK();
    });


/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class Nccl2AllgatherOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2AllgatherOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    loadTypeConfig(c, data_type, reduce_type);
    OP_REQUIRES_OK(c, c->GetAttr("node_size", &node_size));
  }

  ~Nccl2AllgatherOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      auto result_shape = c->input(i).shape();
      result_shape.set_dim(0, result_shape.dim_size(0) * node_size);

      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, result_shape, &output), done);
      CHECK_EQ(ncclSuccess, ncclAllGather((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), data_type, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  int node_size;
  ncclDataType_t data_type;
  ncclRedOp_t reduce_type;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2AllgatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Allgather").Device(DEVICE_GPU), Nccl2AllgatherOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Allgather")
    .Input("tensor: N * T")
    .Output("sum: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("node_size: int")
    .Attr("data_type: string")
    .Attr("reduce_type: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i) {
        c->set_output(i, c->UnknownShape());
      }
      return Status::OK();
    });

/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class Nccl2BroadcastOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2BroadcastOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    loadTypeConfig(c, data_type, reduce_type);

    OP_REQUIRES_OK(c, c->GetAttr("sourceRank", &sourceRank));
    initializeNccl2();
  }

  ~Nccl2BroadcastOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    // se::Stream* tensor_stream = c->op_device_context()->stream();
    // const cudaStream_t cu_stream = reinterpret_cast<const cudaStream_t>(
    //     ((se::cuda::CUDAStream*)tensor_stream->implementation())->cuda_stream());
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      CHECK_EQ(ncclSuccess, ncclBroadcast((const void*)c->input(i).tensor_data().data(), (void*)c->input(i).tensor_data().data(), c->input(i).NumElements(), ncclFloat, sourceRank, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  ncclDataType_t data_type;
  ncclRedOp_t reduce_type;

  int sourceRank;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2BroadcastOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Broadcast").Device(DEVICE_GPU), Nccl2BroadcastOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Broadcast")
    .Input("tensor: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("sourceRank: int")
    .Attr("data_type: string")
    .Attr("reduce_type: string")
    .SetIsStateful();

/////////////////////////////////////////////////////////////////////////////////////
}
}  // namespace tensorflow
