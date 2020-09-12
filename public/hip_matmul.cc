// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_HCC__
#include <hip/hcc_detail/hip_fp16.h>
#include <rocblas.h>
#else
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <vector>

static long __mat_N, __mat_K, __mat_M;
static std::string __mat_P;
static float *A, *B, *C;

#ifndef __HIP_PLATFORM_HCC__
#define __half half
#define rocblas_operation_none CUBLAS_OP_N
#define rocblas_operation_transpose  CUBLAS_OP_T
#define rocblas_handle cublasHandle_t
#define rocblas_create_handle cublasCreate
#endif

namespace {
  hipEvent_t hStart, hEnd;
  hipEvent_t hLeft, hRight;
  float ms; const int loop_runs = 50;
  float *d_m[3];
  rocblas_handle hCublas;
}


template <class F1, class F2, class F3, class F4>
static double compute(const char *type, F1 init, F2 func, F3 zero, F4 cast) {

  printf("======== For %s: ======== (type_size = %zd)\n", type, sizeof(F3));
  init();
  // assert(0 == hipMemset((hipDeviceptr_t)d_m[2], 0, (__mat_N * __mat_M) * sizeof(F3)));
  func();
  assert(0 == hipStreamSynchronize(0));
  assert(0 == hipMemcpyDtoH(C, (hipDeviceptr_t)d_m[2], (__mat_N * __mat_M) * sizeof(F3)));
  assert(0 == hipStreamSynchronize(0));

  F3* h_C = (F3*)C;

  if (1) {
    if (__mat_N * __mat_M > 6) {
      printf("results = [%.6g, %.6g, %.6g, .. %.6g, %.6g, %.6g]\n",
        cast(h_C[0]), cast(h_C[1]), cast(h_C[2]), cast(h_C[__mat_N * __mat_M - 3]), cast(h_C[__mat_N * __mat_M - 2]), cast(h_C[__mat_N * __mat_M - 1]));
    } else {
      printf("results = [");
      for (int i = 0; i < __mat_N * __mat_M; ++i) {
        if (i)
          printf(", ");
        printf("%.6g", cast(h_C[i]));
      }
      printf("]\n");
    }
  }

  double digest = 0;
  for (long i = 0; i < __mat_M * __mat_N; ++i)
    digest += ((i + 1) % 83) * cast(h_C[i]);

  assert(0 == hipEventRecord(hStart, 0));
  for (int i = 0; i < loop_runs; ++i)
    func();
  assert(0 == hipEventRecord(hEnd, 0));

  assert(0 == hipStreamSynchronize(0));
  assert(0 == hipEventElapsedTime(&ms, hStart, hEnd));

  double gflops = 2LU * __mat_N * __mat_K * __mat_M * loop_runs / ms * 1e-6;
  printf(">> (%s) N=%zd, K=%zd, M=%zd, P=%s: %g gflops (digest_v2 = %g), time = %g ms.\n", type, __mat_N, __mat_K, __mat_M, __mat_P.c_str(), gflops, digest, ms / loop_runs);
  return gflops;
}

int main() {
  __mat_N = getenv("N") ? atol(getenv("N")) : 1024;
  __mat_M = getenv("M") ? atol(getenv("M")) : 4096;
  __mat_K = getenv("K") ? atol(getenv("K")) : 64;
  __mat_P = getenv("P") ? std::string(getenv("P")) : "NN";

  // A = new float[__mat_N * __mat_K];
  // B = new float[__mat_M * __mat_K];
  C = new float[__mat_N * __mat_M];

  printf("matmul for N=%zd, K=%zd, M=%zd, P=%s\n", __mat_N, __mat_K, __mat_M, __mat_P.c_str());

  assert(0 == hipInit(0));
  assert(0 == hipSetDevice(0));
  assert(0 == hipMalloc((void**)&d_m[0], (__mat_N * __mat_K) * sizeof(float)));
  assert(0 == hipMalloc((void**)&d_m[1], (__mat_M * __mat_K) * sizeof(float)));
  assert(0 == hipMalloc((void**)&d_m[2], (__mat_N * __mat_M) * sizeof(float)));

  assert(0 == hipEventCreate(&hStart));
  assert(0 == hipEventCreate(&hEnd));

#undef MAT_DATA_TYPE
#undef MAT_BLAS_FUNC
#define MAT_DATA_TYPE float
#ifdef __HIP_PLATFORM_HCC__
#define MAT_BLAS_FUNC rocblas_sgemm
#else
#define MAT_BLAS_FUNC cublasSgemm
#endif

  compute("rocblas_fp32", [&]() {
    if (!hCublas)
      assert(0 == rocblas_create_handle(&hCublas));

    std::vector<float> L(__mat_N * __mat_K), R(__mat_K * __mat_M);
    for (int i = 0; i < L.size(); ++i) L[i] = (i + 1) % 71;
    for (int i = 0; i < R.size(); ++i) R[i] = (i + 2) % 71;

    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[0], L.data(), L.size() * sizeof(float)));
    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[1], R.data(), R.size() * sizeof(float)));
  }, [&]() {
    float alpha = 1.0f, beta = 0.0f;
    if (__mat_P == "NN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_M, (MAT_DATA_TYPE*)d_m[0], __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "TN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_M, (MAT_DATA_TYPE*)d_m[0], __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "NT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_K, (MAT_DATA_TYPE*)d_m[0], __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "TT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_K, (MAT_DATA_TYPE*)d_m[0], __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else
      assert(0);
  }, 0.0f, [](float val) -> double { return double(val); });


#undef MAT_DATA_TYPE
#undef MAT_BLAS_FUNC 
#ifdef __HIP_PLATFORM_HCC__
#define MAT_DATA_TYPE rocblas_half
#define MAT_BLAS_FUNC rocblas_hgemm
#else
#define MAT_DATA_TYPE half
#define MAT_BLAS_FUNC cublasHgemm
#endif

  compute("rocblas_fp16", [&]() {
    if (!hCublas)
      assert(0 == rocblas_create_handle(&hCublas));

    std::vector<__half> L(__mat_N * __mat_K), R(__mat_K * __mat_M);
    for (int i = 0; i < L.size(); ++i) L[i] = __float2half(1);
    for (int i = 0; i < R.size(); ++i) R[i] = __float2half(1);

    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[0], L.data(), L.size() * sizeof(__half)));
    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[1], R.data(), R.size() * sizeof(__half)));
  }, [&]() {
#ifdef __HIP_PLATFORM_HCC__
    __half alpha(1), beta(0);
#else
    __half alpha, beta;
#endif
    if (__mat_P == "NN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_M, (MAT_DATA_TYPE*)d_m[0], __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "TN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_M, (MAT_DATA_TYPE*)d_m[0], __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "NT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_K, (MAT_DATA_TYPE*)d_m[0], __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else if (__mat_P == "TT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], __mat_K, (MAT_DATA_TYPE*)d_m[0], __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], __mat_M));
    } else
      assert(0);
  }, __half(0.0f), [](__half val) -> double { return __half2float(val); });

#ifdef __HIP_PLATFORM_HCC__
#undef MAT_DATA_TYPE
#undef MAT_BLAS_FUNC 
#define MAT_DATA_TYPE int
#define MAT_BLAS_FUNC rocblas_gemm_ex

  (__mat_K % 4 == 0) && compute("rocblas_int8_int32", [&]() {
    if (!hCublas)
      assert(0 == rocblas_create_handle(&hCublas));

    std::vector<unsigned char> L(__mat_N * __mat_K), R(__mat_K * __mat_M);
    for (int i = 0; i < L.size(); ++i) L[i] = 0x01;
    for (int i = 0; i < R.size(); ++i) R[i] = 0x01;

    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[0], L.data(), L.size() * sizeof(unsigned char)));
    assert(0 == hipMemcpyHtoD((hipDeviceptr_t)d_m[1], R.data(), R.size() * sizeof(unsigned char)));
  }, [&]() {
    unsigned alpha = 1, beta = 0;

    if (__mat_P == "NN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], rocblas_datatype_i8_r, __mat_M, (MAT_DATA_TYPE*)d_m[0], rocblas_datatype_i8_r, __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
    } else if (__mat_P == "TN") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_none, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], rocblas_datatype_i8_r, __mat_M, (MAT_DATA_TYPE*)d_m[0], rocblas_datatype_i8_r, __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
    } else if (__mat_P == "NT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_none, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], rocblas_datatype_i8_r, __mat_K, (MAT_DATA_TYPE*)d_m[0], rocblas_datatype_i8_r, __mat_K, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
    } else if (__mat_P == "TT") {
      assert(0 == MAT_BLAS_FUNC(hCublas, rocblas_operation_transpose, rocblas_operation_transpose, __mat_M, __mat_N, __mat_K, (MAT_DATA_TYPE*)&alpha, (MAT_DATA_TYPE*)d_m[1], rocblas_datatype_i8_r, __mat_K, (MAT_DATA_TYPE*)d_m[0], rocblas_datatype_i8_r, __mat_N, (MAT_DATA_TYPE*)&beta, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, (MAT_DATA_TYPE*)d_m[2], rocblas_datatype_i32_r, __mat_M, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0));
    } else
      assert(0);
  }, int(0), [](int val) -> double { return double(val); });
#endif

  return 0;
}
