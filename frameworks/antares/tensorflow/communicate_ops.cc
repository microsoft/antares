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
#define cudaEventElapsedTime hipEventElapsedTime
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

inline void loadTypeConfig(OpKernelConstruction* c, ncclRedOp_t &reduce_type) {
  std::string _reduce_type;
  OP_REQUIRES_OK(c, c->GetAttr("reduce_type", &_reduce_type));

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

inline ncclDataType_t get_nccl_type(DataType dtype) {
  if (dtype == DT_FLOAT)
    return ncclFloat32;
  else if (dtype == DT_INT32)
    return ncclInt32;
  else
    throw std::runtime_error(("Unhandled TF-DataType for communication: " + std::to_string((int)dtype)).c_str());
}

/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class Nccl2AllreduceOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2AllreduceOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    loadTypeConfig(c, reduce_type);
    LOG(INFO) << "Antares Nccl2AllreduceOpKernel Appended.";
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
      CHECK_EQ(ncclSuccess, ncclAllReduce((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), get_nccl_type(output->dtype()), reduce_type, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  ncclRedOp_t reduce_type;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2AllreduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Allreduce").Device(DEVICE_GPU), Nccl2AllreduceOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Allreduce")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
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
    loadTypeConfig(c, reduce_type);
    OP_REQUIRES_OK(c, c->GetAttr("node_size", &node_size));
    LOG(INFO) << "Antares Nccl2ReducescatterOpKernel Appended.";
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
      CHECK_EQ(ncclSuccess, ncclReduceScatter((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements() / node_size, get_nccl_type(output->dtype()), reduce_type, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  int node_size;
  ncclRedOp_t reduce_type;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2ReducescatterOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Reducescatter").Device(DEVICE_GPU), Nccl2ReducescatterOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Reducescatter")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("node_size: int")
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
    OP_REQUIRES_OK(c, c->GetAttr("node_size", &node_size));
    LOG(INFO) << "Antares Nccl2AllgatherOpKernel Appended.";
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
      CHECK_EQ(ncclSuccess, ncclAllGather((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), get_nccl_type(output->dtype()), ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  int node_size;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2AllgatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Allgather").Device(DEVICE_GPU), Nccl2AllgatherOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Allgather")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("node_size: int")
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

    OP_REQUIRES_OK(c, c->GetAttr("source_rank", &source_rank));
    LOG(INFO) << "Antares Nccl2BroadcastOpKernel Appended.";
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
      CHECK_EQ(ncclSuccess, ncclBroadcast((const void*)c->input(i).tensor_data().data(), (void*)c->input(i).tensor_data().data(), c->input(i).NumElements(), get_nccl_type(c->input(i).dtype()), source_rank, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  ncclRedOp_t reduce_type;

  int source_rank;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2BroadcastOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Broadcast").Device(DEVICE_GPU), Nccl2BroadcastOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Broadcast")
    .Input("tensor: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("source_rank: int")
    .SetIsStateful();

/////////////////////////////////////////////////////////////////////////////////////
static cudaEvent_t lastMetricEvent = NULL;

template <typename Device>
class MetricOpKernel: public AsyncOpKernel {
 public:
  explicit MetricOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
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

    auto compute = [&]() {
      for (int i = c->num_inputs() - 1; i >= 0; --i) {
        Tensor* output;
        OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, c->input(i).shape(), &output), done);
        size_t type_size = 0;
        if (output->dtype() == DT_INT32 || output->dtype() == DT_FLOAT)
          type_size = 4;
        CHECK_GT(type_size, 0);
        CHECK_EQ(0, cudaMemcpyAsync((void*)output->tensor_data().data(), (const void*)c->input(i).tensor_data().data(), output->NumElements() * type_size, cudaMemcpyDeviceToDevice, cu_stream));
      }
    };

    cudaEvent_t currMetricEvent;
    CHECK_EQ(0, cudaEventCreateWithFlags(&currMetricEvent, 0));

    pthread_mutex_lock(&__g_lock);
    if (lastMetricEvent) {
      CHECK_EQ(0, cudaEventRecord(currMetricEvent, cu_stream));
      CHECK_EQ(0, cudaStreamSynchronize(cu_stream));
      float ms;
      CHECK_EQ(0, cudaEventElapsedTime(&ms, lastMetricEvent, currMetricEvent));
      if (__ncclComm)
        LOG(INFO) << "Antares Metric Record: ElapsedTime (" << __ncclComm->mpi_rank << "/" << __ncclComm->mpi_size << ") = " << ms * 1e-3 << " sec.";
      else
        LOG(INFO) << "Antares Metric Record: ElapsedTime = " << ms * 1e-3 << " sec.";
      CHECK_EQ(0, cudaEventDestroy(lastMetricEvent));
      CHECK_EQ(0, cudaEventDestroy(currMetricEvent));
      lastMetricEvent = nullptr;
      compute();
    } else {
      LOG(INFO) << "Antares Metric Record: Initialize Metric Record.";
      compute();
      CHECK_EQ(0, cudaEventRecord(currMetricEvent, cu_stream));
      lastMetricEvent = currMetricEvent;
    }
    pthread_mutex_unlock(&__g_lock);

    done();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MetricOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Metric").Device(DEVICE_GPU), MetricOpKernel<GPUDevice>);

REGISTER_OP("Metric")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i)
        c->set_output(i, c->input(i));
      return Status::OK();
    });


/////////////////////////////////////////////////////////////////////////////////////
}
}  // namespace tensorflow
