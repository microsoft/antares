// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn_v7.h>
#include <string>
#include <vector>
#include <unordered_map>

#define ENABLE_DEFAULT_DECLARE
#define SKIP_MAIN_BODY

// #include "convfwd_autotvm.h"

static long __conv_N, __conv_C, __conv_H, __conv_W, __conv_F, __conv_K, __conv_ST, __conv_PD, __conv_D = 1;
static long __conv_HO, __conv_WO;

#ifdef __INT8B_COMPUTE__
#define T int8b
#else
#define T float
#endif

// static T A[__conv_N][__conv_C][__conv_H][__conv_W];
// static T B[__conv_N][__conv_F][__conv_HO][__conv_WO];
// static T K[__conv_F][__conv_C][__conv_K][__conv_K];
static std::vector<T> __A, __B, __K;


namespace {
  cudaEvent_t hStart, hEnd;
  cudaEvent_t hLeft, hRight;
  float ms; const int loop_runs = 100;
  T *d_m[4];
  cudnnHandle_t hCudnn;
}

template <class F1, class F2>
static void compute(const char *type, F1 init, F2 func) {
  printf("======== For %s: ========\n", type);
  if (!init()) {
    printf("CUDNN Conv Algorithm %s doesn't support this compute shape.\n", type);
    return;
  }
  assert(0 == cudaMemset(d_m[1], 0, __B.size()));
  printf("B = %zd\n", __B.size());
  if (!func()) {
    printf("CUDNN Conv Algorithm %s doesn't support this compute shape.\n", type);
    return;
  }
  assert(0 == cudaMemcpy(__B.data(), d_m[1], __B.size(), cudaMemcpyDeviceToHost));
  double dig = 0;
  for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j)
#ifdef __INT8B_COMPUTE__
    printf("%d ", __B[i]);
#else
    printf("%.1f ", __B[i]);
#endif
    puts("...");
  }
  for (int i = 0; i < __B.size(); ++i)
    dig += __B[i] * __B[i];
  printf("... digest_old = %g\n", dig);
  assert(0 == cudaStreamSynchronize(0));

  assert(0 == cudaEventRecord(hStart, 0));
  for (int i = 0; i < loop_runs; ++i)
    func();
  assert(0 == cudaEventRecord(hEnd, 0));

  assert(0 == cudaStreamSynchronize(0));
  assert(0 == cudaEventElapsedTime(&ms, hStart, hEnd));
  printf(">> GFlops = %g\n", 2LU * __conv_N * __conv_HO * __conv_WO * __conv_C * __conv_F * __conv_K * __conv_K * loop_runs / ms * 1e-6);
  printf(">> ms/op = %g\n", ms / loop_runs);
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

  // printf("convfwd for NCHW = (%zd, %zd, %zd, %zd) CO = %zd, K = %zd, S = %zd, P = %zd\n", __conv_N, __conv_C, __conv_H, __conv_W, __conv_F, __conv_K, __conv_ST, __conv_PD);
  assert(0 == cudaSetDevice(0));
  assert(0 == cudnnCreate(&hCudnn));

  cudnnConvolutionDescriptor_t convDesc;
  cudnnTensorDescriptor_t xDesc, yDesc;
  cudnnFilterDescriptor_t kDesc;
  assert(0 == cudnnCreateTensorDescriptor(&xDesc));
  assert(0 == cudnnCreateTensorDescriptor(&yDesc));
  assert(0 == cudnnCreateFilterDescriptor(&kDesc));
  assert(0 == cudnnCreateConvolutionDescriptor(&convDesc));
  int dims[4] = {(int)__conv_N, (int)__conv_C, (int)__conv_H, (int)__conv_W};
  int pad[2] = {(int)__conv_PD, (int)__conv_PD}, fstride[2] = {(int)__conv_ST, (int)__conv_ST}, dila[2] = {(int)__conv_D, (int)__conv_D};
  int oihw[4] = {(int)__conv_F, (int)__conv_C, (int)__conv_K, (int)__conv_K};
  assert(0 == cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2], dims[3]));
  assert(0 == cudnnSetFilter4dDescriptor(kDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, oihw[0], oihw[1], oihw[2], oihw[3]));
  assert(0 == cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, fstride, dila, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  printf("input shape = %d %d %d %d\n", dims[0], dims[1], dims[2], dims[3]);
  assert(0 == cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, kDesc, &dims[0], &dims[1], &dims[2], &dims[3]));
  printf("output shape = %d %d %d %d\n", dims[0], dims[1], dims[2], dims[3]);
  assert(__conv_N == dims[0]);
  assert(__conv_F == dims[1]);

  __conv_HO = dims[2];
  __conv_WO = dims[3];
  assert(0 == cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2], dims[3]));

  __B.resize(__conv_N * __conv_F * __conv_HO * __conv_WO);

#ifdef __INT8B_COMPUTE__
  T in_val = 0x01010101;
#else
  T in_val = 1.0f;
#endif
  __A = std::vector<T>(__conv_N * __conv_C * __conv_H * __conv_W, in_val);
  __K = std::vector<T>(__conv_F * __conv_C * __conv_K * __conv_K, in_val);

  assert(0 == cudaMalloc((void**)&d_m[0], __A.size() * sizeof(T)));
  assert(0 == cudaMalloc((void**)&d_m[1], __B.size() * sizeof(T)));
  assert(0 == cudaMalloc((void**)&d_m[2], __K.size() * sizeof(T)));

  assert(0 == cudaEventCreate(&hStart));
  assert(0 == cudaEventCreate(&hEnd));
  assert(0 == cudaEventCreate(&hLeft));
  assert(0 == cudaEventCreate(&hRight));

  assert(0 == cudaMemcpy(d_m[0], __A.data(), __A.size(), cudaMemcpyHostToDevice));
  assert(0 == cudaMemcpy(d_m[2], __K.data(), __K.size(), cudaMemcpyHostToDevice));

#if 0
  compute("tvm_tune", [&]() {
  }, [&]() {
    LAUNCH_NAME(0, d_m[0], d_m[2], d_m[1]);
    // __impl__conv2d_fwd_128_3_227_227_96_11_11_4_0_1<<<dim3(1,55,128), dim3(5,1,48), 0, 0>>>(d_m[0], d_m[2], d_m[1]);
	  return true;
  });
#endif

  std::vector<cudnnConvolutionFwdAlgo_t> algos = {
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
  };
  std::unordered_map<cudnnConvolutionFwdAlgo_t, std::string> algo_names = {
    {CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"},
    {CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"},
    {CUDNN_CONVOLUTION_FWD_ALGO_GEMM, "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"},
    {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"},
  };

  for (auto algo: algos) {
    size_t ws_size;

    compute(("cudnn::" + algo_names[algo]).c_str(), [&]() {
      if (cudnnGetConvolutionForwardWorkspaceSize(hCudnn, xDesc, kDesc, convDesc, yDesc, algo, &ws_size) != 0)
        return false;
      printf("workspace size = %zd\n", ws_size);
      if (d_m[3]) {
        assert(0 == cudaFree(d_m[3]));
        d_m[3] = NULL;
      }
      if (ws_size)
        assert(0 == cudaMalloc((void**)&d_m[3], ws_size));
      return true;
    }, [&]() {
      float alpha = 1.0f, beta = 0.0f;
      if (cudnnConvolutionForward(hCudnn, &alpha, xDesc, d_m[0], kDesc, d_m[2], convDesc, algo, d_m[3], ws_size, &beta, yDesc, d_m[1]) != 0)
        return false;
      return true;
    });
  }

  return 0;
}
