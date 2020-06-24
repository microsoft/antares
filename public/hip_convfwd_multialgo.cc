// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <unistd.h>
#include <stdlib.h>
#include <vector>

#if 1
// #ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hcc_detail/hip_fp16.h>
#include <miopen/miopen.h>

#define cudnnHandle_t miopenHandle_t
#define cudnnStatus_t miopenStatus_t
#define cudnnTensorDescriptor_t miopenTensorDescriptor_t
#define cudnnConvolutionDescriptor_t miopenConvolutionDescriptor_t
#define cudnnFilterDescriptor_t miopenTensorDescriptor_t
#define cudnnDataType_t miopenDataType_t
#define cudnnConvolutionMode_t miopenConvolutionMode_t

#define cudnnCreate miopenCreate
#define cudnnCreateFilterDescriptor cudnnCreateTensorDescriptor
#define cudnnDestroyFilterDescriptor cudnnDestroyTensorDescriptor
#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor

#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_DATA_HALF miopenHalf
#define CUDNN_CROSS_CORRELATION miopenConvolution
#define cudnnGetConvolution2dForwardOutputDim miopenGetConvolutionForwardOutputDim

#define cudnnSetFilter4dDescriptor(desc, type, fmt, k, c, h, w) \
    cudnnSetTensor4dDescriptor(desc, fmt, type, k, c, h, w)


typedef enum {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2
} cudnnTensorFormat_t;

static cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
  if (format != CUDNN_TENSOR_NCHW)
        return miopenStatusNotImplemented;
  return miopenSet4dTensorDescriptor(tensorDesc, dataType, n, c, h, w);
}

static cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
                                const int padA[], const int filterStrideA[], const int dilationA[],
                                cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  if (arrayLength != 2)
    return miopenStatusNotImplemented;
  return miopenInitConvolutionDescriptor(convDesc, mode,
    padA[0], padA[1], filterStrideA[0], filterStrideA[1], dilationA[0], dilationA[1]);
}

const char *algoName[] = {
  "miopenConvolutionFwdAlgoGEMM",
  "miopenConvolutionFwdAlgoDirect",
  "miopenConvolutionFwdAlgoFFT",
  "miopenConvolutionFwdAlgoWinograd",
  "",
  "miopenConvolutionFwdAlgoImplicitGEMM",
  "miopenConvolutionFwdAlgoStaticCompiledGEMM",
};

#if 0
// Reference Algorithm:
typedef enum {
    miopenConvolutionFwdAlgoGEMM         = 0, /*!< GEMM variant */
    miopenConvolutionFwdAlgoDirect       = 1, /*!< Direct convolutions */
    miopenConvolutionFwdAlgoFFT          = 2, /*!< Fast Fourier Transform indirect convolutions */
    miopenConvolutionFwdAlgoWinograd     = 3, /*!< Winograd indirect convolutions */
    miopenConvolutionFwdAlgoImplicitGEMM = 5, /*!< Implicit GEMM convolutions, fp32 only */
    miopenConvolutionFwdAlgoStaticCompiledGEMM = 6, /*!< Static Compiled GEMM convolutions */
} miopenConvFwdAlgorithm_t;
#endif

#else
#include <cuda_runtime_api.h>
#include <cudnn_v7.h>
#endif

static long __conv_N, __conv_C, __conv_H, __conv_W, __conv_F, __conv_K, __conv_ST, __conv_PD;
static long __conv_HO, __conv_WO;

static float *A, *B, *C;
static size_t full_input, full_kernel, full_output;

namespace {
  hipEvent_t hStart, hEnd;
  hipEvent_t hLeft, hRight;
  float ms; const int loop_runs = 50;
  float *d_m[3];
  cudnnHandle_t hCudnn;
}


template <class F1, class F2, class F3, class F4>
static double compute(const char *type, F1 init, F2 func, F3 zero, F4 cast) {

  printf("======== For %s: ======== (type_size = %zd)\n", type, sizeof(F3));
  init();
  assert(0 == hipMemset((hipDeviceptr_t)d_m[2], 0, (full_output) * sizeof(F3)));
  func();
  assert(0 == hipStreamSynchronize(0));
  assert(0 == hipMemcpyDtoH(C, (hipDeviceptr_t)d_m[2], (full_output) * sizeof(F3)));
  assert(0 == hipStreamSynchronize(0));

  F3* h_C = (F3*)C;

  if (1) {
    if (full_output > 6) {
      printf("results = [%.6g, %.6g, %.6g, .. %.6g, %.6g, %.6g]\n",
        cast(h_C[0]), cast(h_C[1]), cast(h_C[2]), cast(h_C[full_output - 3]), cast(h_C[full_output - 2]), cast(h_C[full_output - 1]));
    } else {
      printf("results = [");
      for (int i = 0; i < full_output; ++i) {
        if (i)
          printf(", ");
        printf("%.6g", cast(h_C[i]));
      }
      printf("]\n");
    }
  }

  double digest = 0;
  for (long i = 0; i < full_output; ++i)
    digest += ((i + 1) % 83) * cast(h_C[i]);

  assert(0 == hipEventRecord(hStart, 0));
  for (int i = 0; i < loop_runs; ++i)
    func();
  assert(0 == hipEventRecord(hEnd, 0));

  assert(0 == hipStreamSynchronize(0));
  assert(0 == hipEventElapsedTime(&ms, hStart, hEnd));

  double gflops = 2LU * full_output * __conv_C * __conv_K * __conv_K * loop_runs / ms * 1e-6;
  printf(">> (%s) N=%zd, C=%zd, H=%zd, W=%zd, F=%zd, K=%zd, ST=%zd, PD=%zd: %g gflops (digest_v2 = %g)\n", type, __conv_N, __conv_C, __conv_H, __conv_W, __conv_F, __conv_K, __conv_ST, __conv_PD, gflops, digest);
  return gflops;
}

int main() {
  __conv_N = getenv("N") ? atol(getenv("N")) : 64;
  __conv_C = getenv("C") ? atol(getenv("C")) : 3;
  __conv_H = getenv("H") ? atol(getenv("H")) : 229;
  __conv_W = getenv("W") ? atol(getenv("W")) : 229;
  __conv_F = getenv("F") ? atol(getenv("F")) : 32;
  __conv_K = getenv("K") ? atol(getenv("K")) : 5;
  __conv_ST = getenv("ST") ? atol(getenv("ST")) : 1;
  __conv_PD = getenv("PD") ? atol(getenv("PD")) : 2;

  printf("convfwd for N=%zd, C=%zd, H=%zd, W=%zd, F=%zd, K=%zd, ST=%zd, PD=%zd\n", __conv_N, __conv_C, __conv_H, __conv_W, __conv_F, __conv_K, __conv_ST, __conv_PD);

  assert(0 == hipInit(0));
  assert(0 == hipSetDevice(0));
  assert(0 == cudnnCreate(&hCudnn));

  cudnnConvolutionDescriptor_t convDesc;
  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t kDesc;
  assert(0 == cudnnCreateTensorDescriptor(&xDesc));
  assert(0 == cudnnCreateTensorDescriptor(&yDesc));
  assert(0 == cudnnCreateFilterDescriptor(&kDesc));
  assert(0 == cudnnCreateConvolutionDescriptor(&convDesc));

  auto dtype = CUDNN_DATA_FLOAT;
  if (getenv("FP16"))
    dtype = CUDNN_DATA_HALF;

  int dims[4] = {(int)__conv_N, (int)__conv_C, (int)__conv_H, (int)__conv_W}, ydims[4];
  int pad[2] = {(int)__conv_PD, (int)__conv_PD}, fstride[2] = {(int)__conv_ST, (int)__conv_ST}, dila[2] = {1, 1};
  int oihw[4] = {(int)__conv_F, (int)__conv_C, (int)__conv_K, (int)__conv_K};
  assert(0 == cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, dtype, dims[0], dims[1], dims[2], dims[3]));
  assert(0 == cudnnSetFilter4dDescriptor(kDesc, dtype, CUDNN_TENSOR_NCHW, oihw[0], oihw[1], oihw[2], oihw[3]));
  assert(0 == cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, fstride, dila, CUDNN_CROSS_CORRELATION, dtype));
  printf("input_shape = %d %d %d %d\n", dims[0], dims[1], dims[2], dims[3]);
  assert(0 == cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, kDesc, &ydims[0], &ydims[1], &ydims[2], &ydims[3]));
  printf("output_shape = %d %d %d %d\n", ydims[0], ydims[1], ydims[2], ydims[3]);
  assert(__conv_N == ydims[0]);
  assert(__conv_F == ydims[1]);
  const long __conv_HO = ydims[2];
  const long __conv_WO = ydims[3];
  assert(0 == cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, dtype, ydims[0], ydims[1], ydims[2], ydims[3]));

  full_input = __conv_N * __conv_C * __conv_H * __conv_W;
  full_kernel = __conv_F * __conv_C * __conv_K * __conv_K;
  full_output = __conv_N * __conv_F * __conv_HO * __conv_WO;

  // A = new float[__mat_N * __mat_K];
  // B = new float[__mat_M * __mat_K];
  C = new float[full_output];

  assert(0 == hipMalloc((void**)&d_m[0], (full_input) * sizeof(float)));
  assert(0 == hipMalloc((void**)&d_m[1], (full_kernel) * sizeof(float)));
  assert(0 == hipMalloc((void**)&d_m[2], (full_output) * sizeof(float)));

  assert(0 == hipEventCreate(&hStart));
  assert(0 == hipEventCreate(&hEnd));

  static bool inited = false; static int returnedAlgoCount, fastest = 0;
  static miopenConvAlgoPerf_t perfResults[4];
  static size_t workspace_size_in_bytes = 0;
  static void *workspace_ptr = NULL;
  const float alpha = 1.0f, beta = 0.0f;

  auto emit_fastest = [&]() {
	  fprintf(stderr, "[MIOpen] Convolution running auto-tune for Forward-Data;\n");
	  assert(0 == miopenFindConvolutionForwardAlgorithm(hCudnn,
				xDesc, d_m[0], kDesc, d_m[1], convDesc, yDesc, d_m[2],
				4, &returnedAlgoCount, perfResults, workspace_ptr, workspace_size_in_bytes, false));
	  for (int i = 1; i < returnedAlgoCount; ++i)
		if (perfResults[i].time < perfResults[fastest].time)
		  fastest = i;
	  workspace_size_in_bytes = perfResults[fastest].memory;
	  if (workspace_ptr != NULL)
		assert(0 == hipFree(workspace_ptr));
	  assert(0 == hipMalloc(&workspace_ptr, workspace_size_in_bytes));
	  printf("Fastest ConvFwd algorithm is: %s, requires workspace = %zd B\n", algoName[perfResults[fastest].fwd_algo], workspace_size_in_bytes);
  };

  emit_fastest();

  compute("convfwd_fp32_fastest", [&]() {
    std::vector<float> L(full_input), R(full_kernel);
    for (int i = 0; i < L.size(); ++i) L[i] = (i + 1) % 71;
    for (int i = 0; i < R.size(); ++i) R[i] = (i + 2) % 71;

    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[0], L.data(), L.size() * sizeof(float)));
    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[1], R.data(), R.size() * sizeof(float)));
  }, [&]() {
    assert(0 == miopenConvolutionForward(hCudnn,
        &alpha, xDesc, d_m[0], kDesc, d_m[1], convDesc, perfResults[fastest].fwd_algo,
        &beta, yDesc, d_m[2], workspace_ptr, workspace_size_in_bytes));
  }, 0.0f, [](float val) -> double { return double(val); });

  compute("convfwd_fp32_direct", [&]() {
    std::vector<float> L(full_input), R(full_kernel);
    for (int i = 0; i < L.size(); ++i) L[i] = (i + 1) % 71;
    for (int i = 0; i < R.size(); ++i) R[i] = (i + 2) % 71;

    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[0], L.data(), L.size() * sizeof(float)));
    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[1], R.data(), R.size() * sizeof(float)));
  }, [&]() {
    assert(0 == miopenConvolutionForward(hCudnn,
        &alpha, xDesc, d_m[0], kDesc, d_m[1], convDesc, miopenConvolutionFwdAlgoDirect,
        &beta, yDesc, d_m[2], workspace_ptr, workspace_size_in_bytes));
  }, 0.0f, [](float val) -> double { return double(val); });

  return 0;
}
