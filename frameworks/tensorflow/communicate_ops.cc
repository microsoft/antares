// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"

#if defined(ANTARES_ROCM)
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

#elif defined(ANTARES_CUDA)
#include <cuda_runtime_api.h>
#include <nccl.h>
#elif defined(ANTARES_MCPU) || defined(ANTARES_SYCL)
#else
#error "Cannot detect which tensorflow platform is using: ANTARES_CUDA/ANTARES_ROCM/ANTARES_MCPU/ANTARES_SYCL."
#endif

#if !defined(__linux__)
#error "Only Linux platform is supported at the moment (with CUDA)."
#endif

#include <dirent.h>
#include <mpi.h>
#include <pthread.h>
#include <sys/stat.h>
#include <chrono>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>


namespace tensorflow {
namespace {

using namespace std;

static pthread_mutex_t __g_lock = PTHREAD_MUTEX_INITIALIZER;

class PeerHandle {
 public:
  PeerHandle() {
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    LOG(INFO) << "PeerHandle Initialize: device-rank = " << mpi_rank;

#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
    ncclUniqueId id;
    if (mpi_rank == 0)
      CHECK_EQ(ncclSuccess, ncclGetUniqueId(&id));
    CHECK_EQ(MPI_SUCCESS, MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_EQ(ncclSuccess, ncclGroupStart());
    CHECK_EQ(ncclSuccess, ncclCommInitRank((ncclComm_t*)&comm, mpi_size, id, mpi_rank));
    CHECK_EQ(ncclSuccess, ncclGroupEnd());
   }

  void* getHandle() const {
    return comm;
  }

  ~PeerHandle() {
    CHECK_EQ(ncclSuccess, ncclCommDestroy((ncclComm_t)comm));
#else
  }

  ~PeerHandle() {
#endif
    LOG(INFO) << "PeerHandle Destroy inter-session communication: device-rank = " << mpi_rank;
  }

  int mpi_size, mpi_rank;
  void* comm;
};

static shared_ptr<PeerHandle> __peerComm;

static shared_ptr<PeerHandle> initializePeers() {
  pthread_mutex_lock(&__g_lock);
  if (__peerComm == nullptr)
    __peerComm = make_shared<PeerHandle>();
  pthread_mutex_unlock(&__g_lock);
  return __peerComm;
}

static void finalizePeers() {
  pthread_mutex_lock(&__g_lock);
  if (__peerComm.use_count() <= 1)
    __peerComm = nullptr;
  pthread_mutex_unlock(&__g_lock);
}


/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class CollectiveOpKernel: public AsyncOpKernel {
 public:
  explicit CollectiveOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), peerComm(initializePeers()) {
    std::string op_type;
    OP_REQUIRES_OK(c, c->GetAttr("op_type", &op_type));
    LOG(INFO) << "[" << peerComm->mpi_rank << "/" << peerComm->mpi_size << "] Antares CollectiveOpKernel[" << op_type << "] Initialized.";
    int delim = op_type.find(':');
    CHECK_EQ(true, delim >= 0);

    op_describe = ([](OpKernelConstruction* c, const char *op_extra) -> void* {
      void* op_describe;

#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
      if (*op_extra == '+')
        op_describe = (void*)(long)ncclSum;
      else if (*op_extra == '>')
        op_describe = (void*)(long)ncclMax;
      else if (*op_extra == '<')
        op_describe = (void*)(long)ncclMin;
#else
      if (*op_extra == '+')
        op_describe = (void*)(long)MPI_SUM;
      else if (*op_extra == '>')
        op_describe = (void*)(long)MPI_MAX;
      else if (*op_extra == '<')
        op_describe = (void*)(long)MPI_MIN;
#endif
      else
        op_describe = (void*)std::atol(op_extra);
      return op_describe;
    })(c, op_type.c_str() + delim + 1);

    if (op_type.substr(0, delim) == "all_reduce")
      multiple = 1, divide = 1, op_base = 0;
    else if (op_type.substr(0, delim) == "reduce_scatter")
      multiple = 1, divide = peerComm->mpi_size, op_base = 1;
    else if (op_type.substr(0, delim) == "all_gather")
      multiple = peerComm->mpi_size, divide = 1, op_base = 2;
    else
      throw std::runtime_error(("Unrecognized collective op_type: " + op_type).c_str());
  }

  ~CollectiveOpKernel() {
    peerComm = nullptr;
    finalizePeers();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
    cudaStream_t cu_stream = *CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(c->op_device_context()->stream()->implementation()->GpuStreamMemberHack()));

    auto dtypeToNative = [](DataType dtype) -> ncclDataType_t {
      switch (dtype) {
        case DT_FLOAT: return ncclFloat32;
        case DT_INT32: return ncclInt32;
        case DT_DOUBLE: return ncclFloat64;
        default:
          throw std::runtime_error(("Unhandled TF-DataType for communication: " + std::to_string((int)dtype)).c_str());
      }
    };

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      CHECK_EQ(0, c->input(i).NumElements() % divide);
      int num_result = c->input(i).NumElements() * multiple / divide;
      auto result_shape = tensorflow::TensorShape({num_result});

      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, result_shape, &output), done);
      switch (op_base) {
        case 0: CHECK_EQ(ncclSuccess, ncclAllReduce    ((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), dtypeToNative(output->dtype()), *(ncclRedOp_t*)&op_describe, (ncclComm_t)peerComm->getHandle(), cu_stream)); break;
        case 1: CHECK_EQ(ncclSuccess, ncclReduceScatter((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), num_result,                dtypeToNative(output->dtype()), *(ncclRedOp_t*)&op_describe, (ncclComm_t)peerComm->getHandle(), cu_stream)); break;
        case 2: CHECK_EQ(ncclSuccess, ncclAllGather    ((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), dtypeToNative(output->dtype()),                           (ncclComm_t)peerComm->getHandle(), cu_stream)); break;
        default:
          throw std::runtime_error(("Unrecognized collective op_base: " + std::to_string(op_base)).c_str());
      }
    }
#else
    auto dtypeToNative = [](DataType dtype) -> MPI_Datatype {
      switch (dtype) {
        case DT_FLOAT: return MPI_FLOAT;
        case DT_INT32: return MPI_INT;
        case DT_DOUBLE: return MPI_DOUBLE;
        default:
          throw std::runtime_error(("Unhandled TF-DataType for communication: " + std::to_string((int)dtype)).c_str());
      }
    };

    std::vector<MPI_Request> requests(c->num_inputs());
    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      CHECK_EQ(0, c->input(i).NumElements() % divide);
      int num_result = c->input(i).NumElements() * multiple / divide;
      auto result_shape = tensorflow::TensorShape({num_result});

      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, result_shape, &output), done);
      auto native_dtype = dtypeToNative(output->dtype());

      switch (op_base) {
        case 0: MPI_Iallreduce((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), native_dtype, *(MPI_Op*)&op_describe, MPI_COMM_WORLD, &requests[i]); break;
        case 1: MPI_Ireduce_scatter((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), std::vector<int>(divide, num_result).data(), native_dtype, *(MPI_Op*)&op_describe, MPI_COMM_WORLD, &requests[i]); break;
        case 2: MPI_Iallgather((const void*)c->input(i).tensor_data().data(), c->input(i).NumElements(), native_dtype, (void*)output->tensor_data().data(), c->input(i).NumElements(), native_dtype, MPI_COMM_WORLD, &requests[i]); break;
        default:
          throw std::runtime_error(("Unrecognized collective op_base: " + std::to_string(op_base)).c_str());
      }
    }

    for (int i = c->num_inputs() - 1; i >= 0; --i)
      MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
#endif
    done();
  }

 private:
  int multiple, divide, op_base;
  void* op_describe;
  shared_ptr<PeerHandle> peerComm;
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveOpKernel);
};

#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
REGISTER_KERNEL_BUILDER(Name("Collective").Device(DEVICE_GPU), CollectiveOpKernel<Eigen::GpuDevice>);
#else
REGISTER_KERNEL_BUILDER(Name("Collective").Device(DEVICE_CPU), CollectiveOpKernel<Eigen::ThreadPoolDevice>);
#endif

REGISTER_OP("Collective")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float64, float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .Attr("op_type: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i)
        c->set_output(i, c->UnknownShape());
      return Status::OK();
    });

/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class SynchronizeOpKernel: public AsyncOpKernel {
 public:
  explicit SynchronizeOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
  }

  ~SynchronizeOpKernel() {
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
    cudaStream_t cu_stream = *CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(c->op_device_context()->stream()->implementation()->GpuStreamMemberHack()));
    CHECK_EQ(cudaSuccess, cudaStreamSynchronize(cu_stream));
#endif
    done();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SynchronizeOpKernel);
};

#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
REGISTER_KERNEL_BUILDER(Name("Synchronize").Device(DEVICE_GPU), SynchronizeOpKernel<Eigen::GpuDevice>);
#else
REGISTER_KERNEL_BUILDER(Name("Synchronize").Device(DEVICE_CPU), SynchronizeOpKernel<Eigen::ThreadPoolDevice>);
#endif

REGISTER_OP("Synchronize")
    .Input("tensor: N * T")
    .Attr("T: {float64, float32, float16, int32, int16, int8}")
    .Attr("N: int >= 1")
    .SetIsStateful();


/////////////////////////////////////////////////////////////////////////////////////
template <typename Device>
class MetricOpKernel: public AsyncOpKernel {
 public:
  explicit MetricOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
    cudaStream_t cu_stream = *CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(c->op_device_context()->stream()->implementation()->GpuStreamMemberHack()));

    static cudaEvent_t lastMetricEvent = NULL;
    auto compute = [&]() {
      for (int i = c->num_inputs() - 1; i >= 0; --i) {
        Tensor* output;
        OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, c->input(i).shape(), &output), done);
        size_t type_size = 0;
        if (output->dtype() == DT_INT32 || output->dtype() == DT_FLOAT)
          type_size = 4;
        else if (output->dtype() == DT_DOUBLE)
          type_size = 8;
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
      if (__peerComm)
        LOG(INFO) << "Antares Metric Record: ElapsedTime (" << __peerComm->mpi_rank << "/" << __peerComm->mpi_size << ") = " << ms * 1e-3 << " sec.";
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
#else
    static std::chrono::time_point<std::chrono::system_clock> lastMetricEvent;
    static bool hasLastEvent = false;

    auto compute = [&]() {
      for (int i = c->num_inputs() - 1; i >= 0; --i) {
        Tensor* output;
        OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, c->input(i).shape(), &output), done);
        size_t type_size = 0;
        if (output->dtype() == DT_INT32 || output->dtype() == DT_FLOAT)
          type_size = 4;
        else if (output->dtype() == DT_DOUBLE)
          type_size = 8;
        CHECK_GT(type_size, 0);
        memcpy((void*)output->tensor_data().data(), (void*)c->input(i).tensor_data().data(), output->NumElements() * type_size);
      }
    };

    pthread_mutex_lock(&__g_lock);
    if (hasLastEvent) {
      auto currMetricEvent = std::chrono::system_clock::now();
      double sec = max(1e-8, 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(currMetricEvent - lastMetricEvent).count());
      LOG(INFO) << "Antares Metric Record: ElapsedTime = " << sec << " sec.";
      hasLastEvent = false;
      compute();
    } else {
      LOG(INFO) << "Antares Metric Record: Initialize Metric Record.";
      compute();
      lastMetricEvent = std::chrono::system_clock::now();
      hasLastEvent = true;
    }
    pthread_mutex_unlock(&__g_lock);
#endif
    done();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MetricOpKernel);
};

#if defined(ANTARES_CUDA) || defined(ANTARES_ROCM)
REGISTER_KERNEL_BUILDER(Name("Metric").Device(DEVICE_GPU), MetricOpKernel<Eigen::GpuDevice>);
#else
REGISTER_KERNEL_BUILDER(Name("Metric").Device(DEVICE_CPU), MetricOpKernel<Eigen::ThreadPoolDevice>);
#endif


REGISTER_OP("Metric")
    .Input("tensor: N * T")
    .Output("result: N * T")
    .Attr("T: {float64, float32, float16, int32, int16, int8}")
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
