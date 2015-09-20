/*************************************************************************************
 * Copyright (c) 2015, Advanced Micro Devices, Inc.  
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this 
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************/

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"

static const clblasOrder order = clblasColumnMajor;
#define pi 3.1415926

namespace caffe {

template <>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
      beta, C, N);
}

template <>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
      beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <>
void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template <>
void caffe_set(const int N, const double alpha, double* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(double) * N);
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

/*
template <>
void caffe_copy<float>(const int N, const float* X, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

template <>
void caffe_copy<double>(const int N, const double* X, double* Y) {
  cblas_dcopy(N, X, 1, Y, 1);
}
*/
template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b, float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b, float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b, float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
	    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
  vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
  vdAbs(n, a, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b, float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b, float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter < Dtype
      > (b, std::numeric_limits < Dtype > ::max());
}
template float caffe_nextafter(const float b);
template double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real < Dtype
      > random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }

  //LOG(INFO) << "caffe_rng_uniform";
}

template void caffe_rng_uniform<float>(const int n, const float a, const float b,
    float* r);
template void caffe_rng_uniform<double>(const int n, const double a, const double b,
    double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a, const Dtype sigma,
    Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution < Dtype > random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void caffe_rng_gaussian<float>(const int n, const float mu, const float sigma,
    float* r);
template void caffe_rng_gaussian<double>(const int n, const double mu,
    const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution < Dtype > random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template void caffe_rng_bernoulli<double>(const int n, const double p, int* r);
template void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution < Dtype > random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> > variate_generator(
      caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);
template void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_dot<float>(const int n, const float* x, const float* y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template <>
double caffe_cpu_dot<double>(const int n, const double* x, const double* y) {
  return cblas_ddot(n, x, 1, y, 1);
}

template <>
int caffe_cpu_hamming_distance<float>(const int n, const float* x,
    const float* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcount(
        static_cast<uint32_t>(x[i]) ^ static_cast<uint32_t>(y[i]));
  }
  return dist;
}

template <>
int caffe_cpu_hamming_distance<double>(const int n, const double* x,
    const double* y) {
  int dist = 0;
  for (int i = 0; i < n; ++i) {
    dist += __builtin_popcountl(
        static_cast<uint64_t>(x[i]) ^ static_cast<uint64_t>(y[i]));
  }
  return dist;
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

INSTANTIATE_CAFFE_CPU_UNARY_FUNC (sign);
INSTANTIATE_CAFFE_CPU_UNARY_FUNC (sgnbit);
INSTANTIATE_CAFFE_CPU_UNARY_FUNC (fabs);

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
    float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
    double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

#ifndef CPU_ONLY
//DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
//  - (x[index] < Dtype(0)));
//DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasSgemm(amdDevice.col, transB, transA, N, M, K, (cl_float) alpha,
          (cl_mem) B, 0, ldb, (cl_mem) A, 0, lda, (cl_float) beta, (cl_mem) C,
          0, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasDgemm(amdDevice.col, transB, transA, N, M, K,  alpha,
          (cl_mem) B, 0, ldb, (cl_mem) A, 0, lda,  beta, (cl_mem) C,
          0, ldc, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
cl_event caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const int offA, const float* B,
    const int offB, const float beta, float* C, const int offC) {
  cl_event event;
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasSgemm(amdDevice.col, transB, transA, N, M, K, (cl_float) alpha,
          (cl_mem) B, offB, ldb, (cl_mem) A, offA, lda, (cl_float) beta,
          (cl_mem) C, offC, ldc, 1, &(amdDevice.CommandQueue), 0, NULL,
          &event));
  return event;
}

template <>
cl_event caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const int offA, const double* B,
    const int offB, const double beta, double* C, const int offC) {
  cl_event event;
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasDgemm(amdDevice.col, transB, transA, N, M, K, alpha,
          (cl_mem) B, offB, ldb, (cl_mem) A, offA, lda, beta,
          (cl_mem) C, offC, ldc, 1, &(amdDevice.CommandQueue), 0, NULL,
          &event));
  return event;
}

template <>
cl_event caffe_gpu_gemm<float>(cl_command_queue *queue,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
    const int N, const int K, const float alpha, const float* A, const int offA,
    const float* B, const int offB, const float beta, float* C,
    const int offC) {
  cl_event event;
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasSgemm(amdDevice.col, transB, transA, N, M, K, (cl_float) alpha,
          (cl_mem) B, offB, ldb, (cl_mem) A, offA, lda, (cl_float) beta,
          (cl_mem) C, offC, ldc, 1, queue, 0, NULL, &event));
  return event;
}

template <>
cl_event caffe_gpu_gemm<double>(cl_command_queue *queue,
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
    const int N, const int K, const double alpha, const double* A,
    const int offA, const double* B, const int offB, const double beta,
    double* C, const int offC) {
  cl_event event;
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  clblasTranspose transB =
      (TransB == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  int ldc = N;
  CLBLAS_CHECK(
      clblasDgemm(amdDevice.col, transB, transA, N, M, K,  alpha,
          (cl_mem) B, offB, ldb, (cl_mem) A, offA, lda, beta,
          (cl_mem) C, offC, ldc, 1, queue, 0, NULL, &event));
  return event;
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, size_t offA, int lda,
    const float* x, size_t offx, const float beta, int incx, float* y,
    size_t offy, int incy) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  CLBLAS_CHECK(
      clblasSgemv(amdDevice.row, transA, M, N, (cl_float) alpha, (cl_mem) A,
          offA, lda, (cl_mem) x, offx, incx, (cl_float) beta, (cl_mem) y, offy,
          incy, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, size_t offA, int lda,
    const double* x, size_t offx, const double beta, int incx, double* y,
    size_t offy, int incy) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  CLBLAS_CHECK(
      clblasDgemv(amdDevice.row, transA, M, N, (cl_double) alpha, (cl_mem) A,
          offA, lda, (cl_mem) x, offx, incx, (cl_double) beta, (cl_mem) y, offy,
          incy, 1, &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  CLBLAS_CHECK(
      clblasSgemv(amdDevice.row, transA, M, N, (cl_float) alpha, (cl_mem) A, 0,
          N, (cl_mem) x, 0, 1, (cl_float) beta, (cl_mem) y, 0, 1, 1,
          &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  clblasTranspose transA =
      (TransA == CblasNoTrans) ? clblasNoTrans : clblasTrans;
  CLBLAS_CHECK(
      clblasDgemv(amdDevice.row, transA, M, N, (cl_double) alpha, (cl_mem) A, 0,
          N, (cl_mem) x, 0, 1, (cl_double) beta, (cl_mem) y, 0, 1, 1,
          &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CLBLAS_CHECK(
      clblasSaxpy(N, alpha, (cl_mem) X, 0, 1, (cl_mem) Y, 0, 1, 1,
          &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CLBLAS_CHECK(
      clblasDaxpy(N, alpha, (cl_mem) X, 0, 1, (cl_mem) Y, 0, 1, 1,
          &(amdDevice.CommandQueue), 0, NULL, NULL));
}

template <>
void caffe_gpu_sgnbit<float>(const int n, const float* x, float* y) {
  caffe_gpu_signbit(n, x, y);
}

template <>
void caffe_gpu_sgnbit<double>(const int n, const double* x, double* y) {
  caffe_gpu_signbit(n, x, y);
}

template <>
void caffe_gpu_abs<float>(const int n, const float* x, float* y) {
  caffe_gpu_abs_ocl(n, x, y);
}

template <>
void caffe_gpu_abs<double>(const int n, const double* x, double* y) {
  caffe_gpu_abs_ocl(n, x, y);
}

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y) {
  clEnqueueReadBuffer(amdDevice.CommandQueue, (cl_mem) X, CL_TRUE, 0, N, Y, 0,
      NULL, NULL);
}
template <>
void caffe_gpu_memcpy<float>(const size_t N, const float* X, float* Y) {
  OCL_CHECK(
      clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem) X, (cl_mem) Y, 0, 0,
          N, 0, NULL, NULL));
}

template <>
void caffe_gpu_memcpy<double>(const size_t N, const double* X, double* Y) {
  OCL_CHECK(
      clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem) X, (cl_mem) Y, 0, 0,
          N, 0, NULL, NULL));
}

template <typename Dtype>
void caffe_gpu_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
     OCL_CHECK(
       clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem) X, (cl_mem) Y, 0, 0,
          N * sizeof(Dtype), 0, NULL, NULL));
  }
}
template void caffe_gpu_copy<float>(const int N, const float* X, float* Y);
template void caffe_gpu_copy<double>(const int N, const double* X, double* Y);
template void caffe_gpu_copy<int>(const int N, const int* X, int* Y);
template void caffe_gpu_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y);

template <>
void caffe_gpu_copy<float>(const int N, const float* X, const int offx, float* Y, const int offy) {
  if (X != Y) {
    CLBLAS_CHECK(
        clblasScopy(N, (cl_mem) X, offx, 1, (cl_mem) Y, offy, 1, 1,
            &(amdDevice.CommandQueue), 0, NULL, NULL));
  }
}

template <>
void caffe_gpu_copy<double>(const int N, const double* X, const int offx, double* Y, const int offy) {
  if (X != Y) {
    CLBLAS_CHECK(
        clblasDcopy(N, (cl_mem) X, offx, 1, (cl_mem) Y, offy, 1, 1,
            &(amdDevice.CommandQueue), 0, NULL, NULL));
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X, const int offx) {
  CLBLAS_CHECK(
      clblasSscal(N, alpha, (cl_mem) X, offx, 1, 1, &(amdDevice.CommandQueue), 0,
          NULL, NULL));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X, const int offx) {
  CLBLAS_CHECK(
      clblasDscal(N, alpha, (cl_mem) X, offx, 1, 1, &(amdDevice.CommandQueue), 0,
          NULL, NULL));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(float)), NULL, NULL);
  cl_mem d_out = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(float)), NULL, NULL);
  clblasSdot(n, d_out, 0, (cl_mem) x, 0, 1, (cl_mem) y, 0, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_out, CL_TRUE, 0, sizeof(float),
      out, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_out);
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  //need to pass in scratchBuff
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(double)), NULL, NULL);
  cl_mem d_out = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(double)), NULL, NULL);
  clblasDdot(n, d_out, 0, (cl_mem) x, 0, 1, (cl_mem) y, 0, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_out, CL_TRUE, 0, sizeof(double),
      out, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_out);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, size_t offx, const float* y, size_t offy, float* out) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(float)), NULL, NULL);
  cl_mem d_out = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(float)), NULL, NULL);
  clblasSdot(n, d_out, 0, (cl_mem) x, offx, 1, (cl_mem) y, offy, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_out, CL_TRUE, 0, sizeof(float),
      out, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_out);
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, size_t offx, const double* y, size_t offy, double * out) {
  //need to pass in scratchBuff
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(double)), NULL, NULL);
  cl_mem d_out = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(double)), NULL, NULL);
  clblasDdot(n, d_out, 0, (cl_mem) x, offx, 1, (cl_mem) y, offy, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_out, CL_TRUE, 0, sizeof(double),
      out, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_out);
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(cl_float)), NULL, NULL);
  cl_mem d_y = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(cl_float)), NULL, NULL);
  clblasSasum(n, d_y, 0, (cl_mem) x, 0, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_y, CL_TRUE, 0, sizeof(float), y,
      0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_y);
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(cl_double)), NULL, NULL);
  cl_mem d_y = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(cl_double)), NULL, NULL);
  clblasDasum(n, d_y, 0, (cl_mem) x, 0, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_y, CL_TRUE, 0, sizeof(double),
      y, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_y);
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, size_t offx, float* y) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(cl_float)), NULL, NULL);
  cl_mem d_y = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(cl_float)), NULL, NULL);
  clblasSasum(n, d_y, 0, (cl_mem) x, offx, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_y, CL_TRUE, 0, sizeof(float), y,
      0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_y);
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, size_t offx, double* y) {
  cl_mem scratchBuff = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (n * sizeof(cl_double)), NULL, NULL);
  cl_mem d_y = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
      (1 * sizeof(cl_double)), NULL, NULL);
  clblasDasum(n, d_y, 0, (cl_mem) x, offx, 1, scratchBuff, 1,
      &(amdDevice.CommandQueue), 0, NULL, NULL);
  clEnqueueReadBuffer(amdDevice.CommandQueue, d_y, CL_TRUE, 0, sizeof(double),
      y, 0, NULL, NULL);
  clReleaseMemObject(scratchBuff);
  clReleaseMemObject(d_y);
}


template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
    float* y) {
  caffe_gpu_copy(n, x, y);
  caffe_gpu_scal(n, alpha, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
    double* y) {
  caffe_gpu_copy(n, x, y);
  caffe_gpu_scal(n, alpha, y);
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
    const int offx, float* y, const int offy) {
  caffe_gpu_copy(n, x, offx, y, offy);
  caffe_gpu_scal(n, alpha, y, offy);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
    const int offx, double* y, const int offy) {
  caffe_gpu_copy(n, x, offx, y, offy);
  caffe_gpu_scal(n, alpha, y, offy);
}

template <typename Dtype>
void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  NOT_IMPLEMENTED;
}

template <>
void caffe_gpu_set<float>(const int N, const float alpha, float* Y, const int offy) {
  ocl_memset(Y, alpha, N, offy);
}

template <>
void caffe_gpu_set<double>(const int N, const double alpha, double* Y, const int offy) {
  ocl_memset(Y, alpha, N, offy);
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  kernel_add_scalar(N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  kernel_add_scalar(N, alpha, Y);
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  kernel_exp(N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  kernel_exp(N, a, y);
}

template <>
void caffe_gpu_sign<float>(const int N, const float *X, float *Y) {
  caffe_gpu_sign_ocl(N, X, Y);
}


template <>
void caffe_gpu_sign<double>(const int N, const double *X, double *Y) {
  caffe_gpu_sign_ocl(N, X, Y);
}

template <>
void caffe_gpu_sign<float>(const int N, const float *X, const int offx, float *Y, const int offy) {
  caffe_gpu_sign_with_offset_ocl(N, X, offx, Y, offy);
}


template <>
void caffe_gpu_sign<double>(const int N, const double *X, const int offx, double *Y, const int offy) {
  caffe_gpu_sign_with_offset_ocl(N, X, offx, Y, offy);
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_sub(N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_sub(N, a, b, y);
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b,
    float* y) {
  kernel_mul(N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a, const double* b,
    double* y) {
  kernel_mul(N, a, b, y);
}

template <>
void caffe_gpu_div<float>(const int N, const float* a, const float* b,
    float* y) {
  kernel_div(N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a, const double* b,
    double* y) {
  kernel_div(N, a, b, y);
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a, const float alpha,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_powx(N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a, const double alpha,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_powx(N, a, alpha, y);
}

void popc_kernel(const int n, const float* a, const float* b, uint8_t* y) {
  NOT_IMPLEMENTED;
}

void popcll_kernel(const int n, const double* a, const double* b, uint8_t* y) {
  NOT_IMPLEMENTED;
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
    const float* y) {
  NOT_IMPLEMENTED;
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
    const double* y) {
  NOT_IMPLEMENTED;
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
        caffe_gpu_uniform(n, r);
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
    float* r) {
  caffe_gpu_uniform(r, n, a, b);	// r is a cl_mem object
}
template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
    double* r) {
  caffe_gpu_uniform(r, n, a, b);  // r is a cl_mem object
}

template <>
void caffe_gpu_rng_gaussian<float>(const int n, const float mu,
    const float sigma, float* r) {
  caffe_gpu_gaussian(r, n, mu, sigma);  // r is a cl_mem object
}

template <>
void caffe_gpu_rng_gaussian<double>(const int n, const double mu,
    const double sigma, double* r) {
  caffe_gpu_gaussian(r, n, mu, sigma);  // r is a cl_mem object
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_log(N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_log(N, a, y);
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_add(N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_add(N, a, b, y);
}
#endif
}  // namespace caffe
