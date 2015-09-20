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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/util/ocl_util.hpp"
#include "caffe/util/ocl_wrapper.hpp"
namespace caffe {

#ifndef CPU_ONLY
typedef unsigned int uint32_t;
struct array4x32 {
    uint32_t v[4];
};
template <typename Dtype>
void caffe_gpu_bernoulli(int* a, const unsigned int n, Dtype inf, Dtype sup,
    Dtype threshold) {
  std::string kernel_name = "RNGBernoulli" + get_dtype_suffix<Dtype>();
  cl_kernel ker_rand = amdDevice.GetKernel(kernel_name);

  static unsigned c = 0;
  unsigned nrounds = 20;
  array4x32 rndctr4;
  rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
  cl_uint size = n / 4; //Note: for correctness, we need to make sure n is dividable by 4

  cl_int ret;
  ret = clSetKernelArg(ker_rand, 0, sizeof(cl_mem), (void*) &a);
  ret |= clSetKernelArg(ker_rand, 1, sizeof(array4x32), (void*) &rndctr4);
  ret |= clSetKernelArg(ker_rand, 2, sizeof(Dtype), (void*) &inf);
  ret |= clSetKernelArg(ker_rand, 3, sizeof(Dtype), (void*) &sup);
  ret |= clSetKernelArg(ker_rand, 4, sizeof(Dtype), (void*) &threshold);
  ret |= clSetKernelArg(ker_rand, 5, sizeof(cl_uint), (void*) &nrounds);
  ret |= clSetKernelArg(ker_rand, 6, sizeof(cl_uint), (void*) &size);
  OCL_CHECK(ret);

  size_t globalws[1] = { size };
  size_t localws[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, ker_rand, 1, NULL,
          globalws, localws, 0, NULL, NULL));
}
template void caffe_gpu_bernoulli<float>(int* a, const unsigned int n,
    float inf, float sup, float threshold);
template void caffe_gpu_bernoulli<double>(int* a, const unsigned int n,
    double inf, double sup, double threshold);

template <typename Dtype>
void transform_gpu(Dtype* src, Dtype* dst, const int top_offset, const int N_,
    const int M_, const int packing_num) {
  std::string kernel_name = "transform" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*) &src);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &dst);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &top_offset);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &N_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &M_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &packing_num);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size2[] = { (size_t)(M_ * packing_num) };
  size_t uiLocal_Work_Size2[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size2, uiLocal_Work_Size2, 0, NULL, NULL));
}

template void transform_gpu<float>(float* src, float* dst, const int top_offset,
    const int N_, const int M_, const int packing_num);
template void transform_gpu<double>(double* src, double* dst,
    const int top_offset, const int N_, const int M_, const int packing_num);

template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* bottom_data, Dtype* scale_data) {
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &bottom_data));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &scale_data));

  size_t Global_Work_Size[1] = { (size_t) num };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void get_max_gpu<float>(cl_kernel Kernel, const int num, const int dim,
    const float* bottom_data, float* scale_data);
template void get_max_gpu<double>(cl_kernel Kernel, const int num,
		const int dim, const double* bottom_data, double* scale_data);

template <typename Dtype>
void caffe_gpu_uniform(Dtype* a, const unsigned int n, Dtype inf, Dtype sup, unsigned int seed_)
{
        static unsigned c = 0;
        if ((n == 0) || (a == NULL)) {
            c = seed_;
            return;
        }
	std::string kernel_name = "RNGUniform" + get_dtype_suffix<Dtype>();
        cl_kernel ker_rand = amdDevice.GetKernel(kernel_name);

        unsigned nrounds = 20;
        array4x32  rndctr4;
        rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
        cl_uint size = n / 4; //Note: for correctness, we need to make sure n is dividable by 4

        cl_int ret;
        ret  = clSetKernelArg(ker_rand, 0, sizeof(cl_mem),     (void*)&a);
        ret |= clSetKernelArg(ker_rand, 1, sizeof(array4x32),  (void*)&rndctr4);
        ret |= clSetKernelArg(ker_rand, 2, sizeof(Dtype),   (void*)&inf);
        ret |= clSetKernelArg(ker_rand, 3, sizeof(Dtype),   (void*)&sup);
        ret |= clSetKernelArg(ker_rand, 4, sizeof(cl_uint),    (void*)&nrounds);
        ret |= clSetKernelArg(ker_rand, 5, sizeof(cl_uint),    (void*)&size);
        OCL_CHECK(ret);

        size_t globalws[1] = {size};
        size_t localws[1] = {256};
        OCL_CHECK (clEnqueueNDRangeKernel(amdDevice.CommandQueue, ker_rand, 1, NULL, globalws, localws, 0, NULL, NULL) );
}
template void caffe_gpu_uniform<float>(float* a, const unsigned int n, float inf, float sup, unsigned int seed_);
template void caffe_gpu_uniform<double>(double* a, const unsigned int n, double inf, double sup, unsigned int seed_);

void caffe_gpu_uniform(const unsigned int n, unsigned int *r, unsigned int _seed)
{
        static unsigned c = 0;
        if ((n == 0) || (r == NULL)) {
            c = _seed;
            return;
        }
        std::string kernel_name = "PRNG_threefry4x32_uint_uniform";
        cl_kernel ker_rand = amdDevice.GetKernel(kernel_name);

        unsigned nrounds = 20;
        array4x32  rndctr4;
        rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
        cl_uint size = n / 4; //Note: for correctness, we need to make sure n is dividable by 4
        
        cl_uint inf = 0;
        cl_uint sup = UINT_MAX;
        cl_int ret;
        ret  = clSetKernelArg(ker_rand, 0, sizeof(cl_mem),     (void*)&r);
        ret |= clSetKernelArg(ker_rand, 1, sizeof(array4x32),  (void*)&rndctr4);
        ret |= clSetKernelArg(ker_rand, 2, sizeof(cl_uint),   (void*)&inf);
        ret |= clSetKernelArg(ker_rand, 3, sizeof(cl_uint),   (void*)&sup);
        ret |= clSetKernelArg(ker_rand, 4, sizeof(cl_uint),    (void*)&nrounds);
        ret |= clSetKernelArg(ker_rand, 5, sizeof(cl_uint),    (void*)&size);
        OCL_CHECK(ret);

        size_t globalws[1] = {size};
        size_t localws[1] = {256};
        OCL_CHECK (clEnqueueNDRangeKernel(amdDevice.CommandQueue, ker_rand, 1, NULL, globalws, localws, 0, NULL, NULL) );
}

template <typename Dtype>
void caffe_gpu_gaussian(Dtype* a, const unsigned int n, Dtype E, Dtype V)
{
        std::string kernel_name = "RNGGaussian" + get_dtype_suffix<Dtype>();
        cl_kernel ker_rand = amdDevice.GetKernel(kernel_name);

        static unsigned c = 0;
        unsigned nrounds = 20;
        array4x32  rndctr4;
        rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
        cl_uint size = n / 4; //Note: for correctness, we need to make sure n is dividable by 4

        cl_int ret;
        ret  = clSetKernelArg(ker_rand, 0, sizeof(cl_mem),     (void*)&a);
        ret |= clSetKernelArg(ker_rand, 1, sizeof(array4x32),  (void*)&rndctr4);
        ret |= clSetKernelArg(ker_rand, 2, sizeof(Dtype),   (void*)&E);
        ret |= clSetKernelArg(ker_rand, 3, sizeof(Dtype),   (void*)&V);
        ret |= clSetKernelArg(ker_rand, 4, sizeof(cl_uint),    (void*)&nrounds);
        ret |= clSetKernelArg(ker_rand, 5, sizeof(cl_uint),    (void*)&size);
        OCL_CHECK(ret);

        size_t globalws[1] = {size};
        size_t localws[1] = {256};
        OCL_CHECK (clEnqueueNDRangeKernel(amdDevice.CommandQueue, ker_rand, 1, NULL, globalws, localws, 0, NULL, NULL) );
}
template void caffe_gpu_gaussian<float>(float* a, const unsigned int n, float E, float V);
template void caffe_gpu_gaussian<double>(double* a, const unsigned int n, double E, double V);

template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out) {
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) num };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void exp_gpu<float>(cl_kernel Kernel, const int num, const float* data,
    float* out);
template void exp_gpu<double>(cl_kernel Kernel, const int num,
    const double* data, double* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &scale));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &data));

  size_t Global_Work_Size[1] = { (size_t)(num * dim) };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void softmax_div_gpu<float>(cl_kernel Kernel, const int num,
    const int dim, const float* scale, float* data);
template void softmax_div_gpu<double>(cl_kernel Kernel, const int num,
    const int dim, const double* scale, double* data);

template <typename Dtype>
Dtype softmax_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* prob_data, const Dtype* label, cl_mem d_loss) {

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*) &prob_data));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &d_loss));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &label));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 5, 256 * sizeof(Dtype), NULL));

  size_t globalws[1] = { 256 };
  size_t localws[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, globalws,
          localws, 0, NULL, NULL));
  void* h_loss = clEnqueueMapBuffer(amdDevice.CommandQueue, d_loss, CL_TRUE,
      CL_MAP_READ, 0, sizeof(Dtype), 0, NULL, NULL, NULL);
  Dtype loss = *(Dtype*) h_loss;
  clEnqueueUnmapMemObject(amdDevice.CommandQueue, d_loss, h_loss, 0, NULL,
      NULL);

  return loss;
}

template float softmax_gpu<float>(cl_kernel Kernel, const int num,
    const int dim, const float* prob_data, const float* label, cl_mem d_loss);
template double softmax_gpu<double>(cl_kernel Kernel, const int num,
    const int dim, const double* prob_data, const double* label, cl_mem d_loss);

template <typename Dtype>
void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  std::string kernel_name = "kernel_channel_max" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &channels));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t)(num * spatial_dim) };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_channel_max<float>(const int num, const int channels,
    const int spatial_dim, const float* data, float* out);
template void kernel_channel_max<double>(const int num, const int channels,
    const int spatial_dim, const double* data, double* out);

template <typename Dtype>
void kernel_channel_subtract(const int count, const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  std::string kernel_name = "kernel_channel_subtract"
      + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &channels));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &channel_max));
  OCL_CHECK(clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &data));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_channel_subtract<float>(const int count, const int num,
    const int channels, const int spatial_dim, const float* channel_max,
    float* data);
template void kernel_channel_subtract<double>(const int count, const int num,
    const int channels, const int spatial_dim, const double* channel_max,
    double* data);

template <typename Dtype>
void kernel_mul(const int count, const Dtype* a, const Dtype* b, Dtype* out) {
  std::string kernel_name = "kernel_mul" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_mul<float>(const int count, const float* a, const float* b,
    float* out);
template void kernel_mul<double>(const int count, const double* a,
    const double* b, double* out);

template <typename Dtype>
void kernel_add_scalar(const int count, const Dtype data, Dtype* out) {
  std::string kernel_name = "kernel_add_scalar" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_add_scalar<float>(const int count, const float data,
    float* out);
template void kernel_add_scalar<double>(const int count, const double data,
    double* out);

template <typename Dtype>
void kernel_powx(const int count, const Dtype* data, const Dtype alpha,
    Dtype* out) {
  std::string kernel_name = "kernel_powx" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(Dtype), (void*) &alpha));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_powx<float>(const int count, const float* data,
    const float alpha, float* out);
template void kernel_powx<double>(const int count, const double* data,
    const double alpha, double* out);

template <typename Dtype>
void kernel_div(const int count, const Dtype* a, const Dtype* b, Dtype* out) {
  std::string kernel_name = "kernel_div" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_div<float>(const int count, const float* a, const float* b,
    float* out);
template void kernel_div<double>(const int count, const double* a,
    const double* b, double* out);

template <typename Dtype>
void kernel_add(const int count, const Dtype* a, const Dtype* b, Dtype* out) {
  std::string kernel_name = "kernel_add" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_add<float>(const int count, const float* a, const float* b,
    float* out);
template void kernel_add<double>(const int count, const double* a,
    const double* b, double* out);

template <typename Dtype>
void kernel_sub(const int count, const Dtype* a, const Dtype* b, Dtype* out) {
  std::string kernel_name = "kernel_sub" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_sub<float>(const int count, const float* a, const float* b,
    float* out);
template void kernel_sub<double>(const int count, const double* a,
    const double* b, double* out);

template <typename Dtype>
void kernel_log(const int count, const Dtype* data, Dtype* out) {
  std::string kernel_name = "kernel_log" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_log<float>(const int count, const float* data, float* out);
template void kernel_log<double>(const int count, const double* data,
    double* out);

template <typename Dtype>
void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  std::string kernel_name = "kernel_exp" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &out));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_exp<float>(const int count, const float* data, float* out);
template void kernel_exp<double>(const int count, const double* data,
    double* out);

template <typename Dtype>
void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  std::string kernel_name = "kernel_channel_sum" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &channels));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &channel_sum));

  size_t Global_Work_Size[1] = { (size_t)(num * spatial_dim) };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_channel_sum<float>(const int num, const int channels,
    const int spatial_dim, const float* data, float* channel_sum);
template void kernel_channel_sum<double>(const int num, const int channels,
    const int spatial_dim, const double* data, double* channel_sum);

template <typename Dtype>
void kernel_channel_div(const int count, const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  std::string kernel_name = "kernel_channel_div" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &channels));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &channel_sum));
  OCL_CHECK(clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &data));

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_channel_div<float>(const int count, const int num,
    const int channels, const int spatial_dim, const float* channel_sum,
    float* data);
template void kernel_channel_div<double>(const int count, const int num,
    const int channels, const int spatial_dim, const double* channel_sum,
    double* data);

template <typename Dtype>
void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  std::string kernel_name = "kernel_channel_dot" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &channels));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &data_1));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &data_2));
  OCL_CHECK(clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &channel_dot));

  size_t Global_Work_Size[1] = { (size_t)(num * spatial_dim) };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void kernel_channel_dot<float>(const int num, const int channels,
    const int spatial_dim, const float* data_1, const float* data_2,
    float* channel_dot);
template void kernel_channel_dot<double>(const int num, const int channels,
    const int spatial_dim, const double* data_1, const double* data_2,
    double* channel_dot);

template <typename Dtype>
void SoftmaxLossForwardGPU(const int nthreads, const Dtype* prob_data,
    const Dtype* label, Dtype* loss, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, Dtype* counts) {
  std::string kernel_name = "SoftmaxLossForwardGPU" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  int int_has_ignore_label = has_ignore_label_ ? 1 : 0;
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &nthreads));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &prob_data));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &label));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &loss));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(
      clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &int_has_ignore_label));
  OCL_CHECK(clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &ignore_label_));
  OCL_CHECK(clSetKernelArg(Kernel, 9, sizeof(cl_mem), (void*) &counts));

  size_t Global_Work_Size[1] = { (size_t) nthreads };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void SoftmaxLossForwardGPU<float>(const int nthreads,
    const float* prob_data, const float* label, float* loss, const int num,
    const int dim, const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, float* counts);
template void SoftmaxLossForwardGPU<double>(const int nthreads,
    const double* prob_data, const double* label, double* loss, const int num,
    const int dim, const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, double* counts);

template <typename Dtype>
void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
    const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, Dtype* counts) {
  std::string kernel_name = "SoftmaxLossBackwardGPU"
      + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  int int_has_ignore_label = has_ignore_label_ ? 1 : 0;

  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &nthreads));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &label));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_diff));
  OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &spatial_dim));
  OCL_CHECK(
      clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &int_has_ignore_label));
  OCL_CHECK(clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &ignore_label_));
  OCL_CHECK(clSetKernelArg(Kernel, 9, sizeof(cl_mem), (void*) &counts));

  size_t Global_Work_Size[1] = { (size_t) nthreads };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void SoftmaxLossBackwardGPU<float>(const int nthreads,
    const float* top, const float* label, float* bottom_diff, const int num,
    const int dim, const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, float* counts);
template void SoftmaxLossBackwardGPU<double>(const int nthreads,
    const double* top, const double* label, double* bottom_diff, const int num,
    const int dim, const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, double* counts);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data) {
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*) &alpha));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &data));

  size_t Global_Work_Size[1] = { (size_t) num };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void scal_gpu<float>(cl_kernel Kernel, const int num,
    const float alpha, float* data);
template void scal_gpu<double>(cl_kernel Kernel, const int num,
    const double alpha, double* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, int dim, Dtype* data,
    const Dtype* label) {
  OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num));
  OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &dim));
  OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &data));
  OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &label));

  size_t Global_Work_Size[1] = { (size_t) num };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void diff_gpu<float>(cl_kernel Kernel, const int num, const int dim,
    float* data, const float* label);
template void diff_gpu<double>(cl_kernel Kernel, const int num, const int dim,
    double* data, const double* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    Dtype* top_data) {
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_size_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &stride_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void max_pool_fp_gpu<float>(cl_kernel Kernel, const int count,
    const float* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    float* top_data);
template void max_pool_fp_gpu<double>(cl_kernel Kernel, const int count,
    const double* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    double* top_data);

template <typename Dtype>
void MaxPoolForward(const int count, const Dtype* bottom_data, const int clnum,
    const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, Dtype* top_data, int* mask,
    Dtype* top_mask) {
  std::string kernel_name = "MaxPoolForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_h_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_w_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w_);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &pad_h_);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &pad_w_);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 15, sizeof(cl_mem), (void*) &mask);
  ret |= clSetKernelArg(Kernel, 16, sizeof(cl_mem), (void*) &top_mask);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void MaxPoolForward<float>(const int count, const float* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, float* top_data, int* mask,
    float* top_mask);
template void MaxPoolForward<double>(const int count, const double* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, double* top_data, int* mask,
    double* top_mask);

template <typename Dtype>
void StoPoolForwardTrain(const int count, const Dtype* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    Dtype* idx_data, Dtype* top_data) {
  std::string kernel_name = "StoPoolForwardTrain" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_h_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_w_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w_);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_mem), (void*) &idx_data);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void StoPoolForwardTrain<float>(const int count,
    const float* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_h_, const int kernel_w_,
    const int stride_h_, const int stride_w_, float* idx_data, float* top_data);
template void StoPoolForwardTrain<double>(const int count,
    const double* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_h_, const int kernel_w_,
    const int stride_h_, const int stride_w_, double* idx_data,
    double* top_data);

template <typename Dtype>
void StoPoolForwardTest(const int count, const Dtype* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    Dtype* top_data) {
  std::string kernel_name = "StoPoolForwardTest" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_h_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_w_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w_);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));

}
template void StoPoolForwardTest<float>(const int count,
    const float* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_h_, const int kernel_w_,
    const int stride_h_, const int stride_w_, float* top_data);
template void StoPoolForwardTest<double>(const int count,
    const double* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_h_, const int kernel_w_,
    const int stride_h_, const int stride_w_, double* top_data);

template <typename Dtype>
void AvePoolForward(const int count, const Dtype* bottom_data, const int clnum,
    const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, Dtype* top_data) {
  std::string kernel_name = "AvePoolForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_h_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_w_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w_);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &pad_h_);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &pad_w_);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void AvePoolForward<float>(const int count, const float* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, float* top_data);
template void AvePoolForward<double>(const int count, const double* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, double* top_data);

template <typename Dtype>
void ave_pool_fp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, Dtype* top_data) {
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_size_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &stride_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &pad_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void ave_pool_fp_gpu<float>(cl_kernel Kernel, const int count,
    const float* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, float* top_data);
template void ave_pool_fp_gpu<double>(cl_kernel Kernel, const int count,
    const double* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, double* top_data);

template <typename Dtype>
void max_pool_bp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_size_,
    const int stride_, Dtype* bottom_diff) {
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &kernel_size_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void max_pool_bp_gpu<float>(cl_kernel Kernel, const int count,
    const float* bottom_data, const float* top_data, const float* top_diff,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_size_,
    const int stride_, float* bottom_diff);
template void max_pool_bp_gpu<double>(cl_kernel Kernel, const int count,
    const double* bottom_data, const double* top_data, const double* top_diff,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_size_,
    const int stride_, double* bottom_diff);

template <typename Dtype>
void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  std::string kernel_name = "MaxPoolBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &mask);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &top_mask);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &num);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pooled_height);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &pooled_width);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &kernel_w);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 15, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 16, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void MaxPoolBackward<float>(const int nthreads,
    const float* const top_diff, const int* const mask,
    const float* const top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff);
template void MaxPoolBackward<double>(const int nthreads,
    const double* const top_diff, const int* const mask,
    const double* const top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    double* const bottom_diff);

template <typename Dtype>
void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  std::string kernel_name = "AvePoolBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &num);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_w);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void AvePoolBackward<float>(const int nthreads,
    const float* const top_diff, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const bottom_diff);
template void AvePoolBackward<double>(const int nthreads,
    const double* const top_diff, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    double* const bottom_diff);

template <typename Dtype>
void StoPoolBackward(const int nthreads, const Dtype* const rand_idx,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, Dtype* const bottom_diff) {
  std::string kernel_name = "StoPoolBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &rand_idx);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &num);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_height);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pooled_width);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &kernel_w);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void StoPoolBackward<float>(const int nthreads,
    const float* const rand_idx, const float* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    float* const bottom_diff);
template void StoPoolBackward<double>(const int nthreads,
    const double* const rand_idx, const double* const top_diff, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w,
    double* const bottom_diff);

template <typename Dtype>
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_size_,
    const int stride_, const int pad_, Dtype* bottom_diff) {
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &clnum);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &channels_);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height_);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width_);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &pooled_height_);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pooled_width_);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &kernel_size_);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &stride_);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &pad_);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void ave_pool_bp_gpu<float>(cl_kernel Kernel, const int count,
    const float* top_diff, const int clnum, const int channels_,
    const int intheight_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, float* bottom_diff);
template void ave_pool_bp_gpu<double>(cl_kernel Kernel, const int count,
    const double* top_diff, const int clnum, const int channels_,
    const int intheight_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, double* bottom_diff);

template <typename Dtype>
void PReLUForward(const int count, const int channels, const int dim,
    const Dtype* bottom_data, Dtype* top_data, const Dtype* slope_data,
    const int div_factor) {
  std::string kernel_name = "PReLUForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &dim);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &slope_data);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &div_factor);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void PReLUForward<float>(const int count, const int channels,
    const int dim, const float* bottom_data, float* top_data,
    const float* slope_data, const int div_factor);
template void PReLUForward<double>(const int count, const int channels,
    const int dim, const double* bottom_data, double* top_data,
    const double* slope_data, const int div_factor);

template <typename Dtype>
void PReLUBackward(const int count, const int channels, const int dim,
    const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff,
    const Dtype* slope_data, const int div_factor) {
  std::string kernel_name = "PReLUBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &dim);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &bottom_diff);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_mem), (void*) &slope_data);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &div_factor);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void PReLUBackward<float>(const int count, const int channels,
    const int dim, const float* top_diff, const float* bottom_data,
    float* bottom_diff, const float* slope_data, const int div_factor);
template void PReLUBackward<double>(const int count, const int channels,
    const int dim, const double* top_diff, const double* bottom_data,
    double* bottom_diff, const double* slope_data, const int div_factor);

template <typename Dtype>
void PReLUParamBackward(const int count, const Dtype* top_diff,
    const int offset_out, const Dtype* bottom_data, const int offset_in,
    Dtype* bottom_diff) {
  std::string kernel_name = "PReLUParamBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret = clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &offset_out);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_data);
  ret = clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &offset_in);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_mem), (void*) &bottom_diff);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void PReLUParamBackward<float>(const int count, const float* top_diff,
    const int offset_out, const float* bottom_data, const int offset_in,
    float* bottom_diff);
template void PReLUParamBackward<double>(const int count,
    const double* top_diff, const int offset_out, const double* bottom_data,
    const int offset_in, double* bottom_diff);

template <typename Dtype>
void ReLUForward(const int count, const Dtype* bottom_data, Dtype* top_data,
    Dtype negative_slope) {
  std::string kernel_name = "ReLUForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(Dtype), (void*) &negative_slope);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void ReLUForward<float>(const int count, const float* bottom_data,
    float* top_data, float negative_slope);
template void ReLUForward<double>(const int count, const double* bottom_data,
    double* top_data, double negative_slope);

template <typename Dtype>
void ReLUBackward(const int count, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype negative_slope) {
  std::string kernel_name = "ReLUBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_diff);
  ret |= clSetKernelArg(Kernel, 4, sizeof(Dtype), (void*) &negative_slope);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void ReLUBackward<float>(const int count, const float* top_diff,
    const float* bottom_data, float* bottom_diff, float negative_slope);
template void ReLUBackward<double>(const int count, const double* top_diff,
    const double* bottom_data, double* bottom_diff, double negative_slope);

template <typename Dtype>
void SigmoidForward(const int count, const Dtype* bottom_data,
    Dtype* top_data) {
  std::string kernel_name = "SigmoidForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void SigmoidForward<float>(const int count, const float* bottom_data,
    float* top_data);
template void SigmoidForward<double>(const int count, const double* bottom_data,
    double* top_data);

template <typename Dtype>
void SigmoidBackward(const int count, const Dtype* top_diff,
    const Dtype* top_data, Dtype* bottom_diff) {
  std::string kernel_name = "SigmoidBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void SigmoidBackward<float>(const int count, const float* top_diff,
    const float* top_data, float* bottom_diff);
template void SigmoidBackward<double>(const int count, const double* top_diff,
    const double* top_data, double* bottom_diff);

template <typename Dtype>
void ThresholdForward(const int count, const Dtype threshold,
    const Dtype* bottom_data, Dtype* top_data) {
  std::string kernel_name = "ThresholdForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*) &threshold);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void ThresholdForward<float>(const int count, const float threshold,
    const float* bottom_data, float* top_data);
template void ThresholdForward<double>(const int count, const double threshold,
    const double* bottom_data, double* top_data);

template <typename Dtype>
void TanHForward(const int count, const Dtype* bottom_data, Dtype* top_data) {
  std::string kernel_name = "TanHForward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void TanHForward<float>(const int count, const float* bottom_data,
    float* top_data);
template void TanHForward<double>(const int count, const double* bottom_data,
    double* top_data);

template <typename Dtype>
void TanHBackward(const int count, const Dtype* top_diff, const Dtype* top_data,
    Dtype* bottom_diff) {
  std::string kernel_name = "TanHBackward" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) count };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void TanHBackward<float>(const int count, const float* top_diff,
    const float* top_data, float* bottom_diff);
template void TanHBackward<double>(const int count, const double* top_diff,
    const double* top_data, double* bottom_diff);

template <typename Dtype>
void opttrans(const Dtype* data_im, const int im_offset, const int channels,
    const int height, const int width, Dtype* data_opt, const int opt_offset,
    const int optnum) {
  std::string kernel_name = "opttrans" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  int num_kernels = channels * height * width * optnum;

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num_kernels);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data_im);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &im_offset);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_mem), (void*) &data_opt);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &opt_offset);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &optnum);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) num_kernels };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void opttrans<float>(const float* data_im, const int im_offset,
    const int channels, const int height, const int width, float* data_opt,
    const int opt_offset, const int optnum);
template void opttrans<double>(const double* data_im, const int im_offset,
    const int channels, const int height, const int width, double* data_opt,
    const int opt_offset, const int optnum);

template <typename Dtype>
void LRNFillScale(const int nthreads, const Dtype* const in, const int num,
    const int channels, const int height, const int width, const int size,
    const Dtype alpha_over_size, const Dtype k, Dtype* const scale) {
  std::string kernel_name = "LRNFillScale" + get_dtype_suffix<Dtype>();
  cl_kernel LFSkernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(LFSkernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(LFSkernel, 1, sizeof(cl_mem), (void*) &in);
  ret |= clSetKernelArg(LFSkernel, 2, sizeof(cl_int), (void*) &num);
  ret |= clSetKernelArg(LFSkernel, 3, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(LFSkernel, 4, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(LFSkernel, 5, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(LFSkernel, 6, sizeof(cl_int), (void*) &size);
  ret |= clSetKernelArg(LFSkernel, 7, sizeof(Dtype), (void*) &alpha_over_size);
  ret |= clSetKernelArg(LFSkernel, 8, sizeof(Dtype), (void*) &k);
  ret |= clSetKernelArg(LFSkernel, 9, sizeof(cl_mem), (void*) &scale);
  OCL_CHECK(ret);
  size_t uiGlobal_Work_Size[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, LFSkernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void LRNFillScale<float>(const int nthreads, const float* const in,
    const int num, const int channels, const int height, const int width,
    const int size, const float alpha_over_size, const float k,
    float* const scale);
template void LRNFillScale<double>(const int nthreads, const double* const in,
    const int num, const int channels, const int height, const int width,
    const int size, const double alpha_over_size, const double k,
    double* const scale);

template <typename Dtype>
void LRNComputeOutput(int nthreads, const Dtype* in, Dtype* scale,
    Dtype negative_beta, Dtype* out) {
  std::string kernel_name = "LRNComputeOutput" + get_dtype_suffix<Dtype>();
  cl_kernel LCOkernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(LCOkernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(LCOkernel, 1, sizeof(cl_mem), (void*) &in);
  ret |= clSetKernelArg(LCOkernel, 2, sizeof(cl_mem), (void*) &scale);
  ret |= clSetKernelArg(LCOkernel, 3, sizeof(Dtype), (void*) &negative_beta);
  ret |= clSetKernelArg(LCOkernel, 4, sizeof(cl_mem), (void*) &out);
  OCL_CHECK(ret);
  size_t uiGlobal_Work_Size2[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size2[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, LCOkernel, 1, NULL,
          uiGlobal_Work_Size2, uiLocal_Work_Size2, 0, NULL, NULL));
}
template void LRNComputeOutput<float>(int nthreads, const float* in,
    float* scale, float negative_beta, float* out);
template void LRNComputeOutput<double>(int nthreads, const double* in,
    double* scale, double negative_beta, double* out);

template <typename Dtype>
void LRNComputeDiff(const int nthreads, const Dtype* const bottom_data,
    const Dtype* const top_data, const Dtype* const scale,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int size,
    const Dtype negative_beta, const Dtype cache_ratio,
    Dtype* const bottom_diff) {
  std::string kernel_name = "LRNComputeDiff" + get_dtype_suffix<Dtype>();
  cl_kernel LCDkernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(LCDkernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(LCDkernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(LCDkernel, 2, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(LCDkernel, 3, sizeof(cl_mem), (void*) &scale);
  ret |= clSetKernelArg(LCDkernel, 4, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(LCDkernel, 5, sizeof(cl_int), (void*) &num);
  ret |= clSetKernelArg(LCDkernel, 6, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(LCDkernel, 7, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(LCDkernel, 8, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(LCDkernel, 9, sizeof(cl_int), (void*) &size);
  ret |= clSetKernelArg(LCDkernel, 10, sizeof(Dtype), (void*) &negative_beta);
  ret |= clSetKernelArg(LCDkernel, 11, sizeof(Dtype), (void*) &cache_ratio);
  ret |= clSetKernelArg(LCDkernel, 12, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);
  size_t uiGlobal_Work_Size[] = { (size_t) nthreads };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, LCDkernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}
template void LRNComputeDiff<float>(const int nthreads,
    const float* const bottom_data, const float* const top_data,
    const float* const scale, const float* const top_diff, const int num,
    const int channels, const int height, const int width, const int size,
    const float negative_beta, const float cache_ratio,
    float* const bottom_diff);
template void LRNComputeDiff<double>(const int nthreads,
    const double* const bottom_data, const double* const top_data,
    const double* const scale, const double* const top_diff, const int num,
    const int channels, const int height, const int width, const int size,
    const double negative_beta, const double cache_ratio,
    double* const bottom_diff);

template <typename Dtype>
void caffe_gpu_add(const int n, const Dtype* in1, const Dtype* in2, Dtype* y) {
  std::string kernel_name = "caffe_gpu_add" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &n);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &in1);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &in2);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) n };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_add<float>(const int n, const float* in1,
    const float* in2, float* y);
template void caffe_gpu_add<double>(const int n, const double* in1,
    const double* in2, double* y);

template <typename Dtype>
void caffe_gpu_signbit(const int N, const Dtype* X, Dtype * Y) {
  std::string kernel_name = "caffe_gpu_sgnbit" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &N);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &X);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &Y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) N };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void caffe_gpu_signbit<float>(const int N, const float* X, float * Y);
template void caffe_gpu_signbit<double>(const int N, const double* X, double * Y);

template <typename Dtype>
void caffe_gpu_sign_ocl(const int N, const Dtype* X, Dtype * Y) {
  std::string kernel_name = "caffe_gpu_sign" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &N);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &X);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &Y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) N };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_sign_ocl<float>(const int N, const float* X, float* Y);
template void caffe_gpu_sign_ocl<double>(const int N, const double* X,
    double* Y);

template <typename Dtype>
void caffe_gpu_sign_with_offset_ocl(const int N, const Dtype* X, const int offx,  Dtype * Y, const int offy) {
  std::string kernel_name = "caffe_gpu_sign_with_offset" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &N);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &X);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &offx);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &Y);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &offy);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) N };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_sign_with_offset_ocl<float>(const int N, const float* X, const int offx, float* Y, const int offy);
template void caffe_gpu_sign_with_offset_ocl<double>(const int N, const double* X, const int offx, double* Y, const int offy);


template <typename Dtype>
void caffe_gpu_abs_ocl(const int N, const Dtype* X, Dtype * Y) {
  std::string kernel_name = "caffe_gpu_abs" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &N);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &X);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &Y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) N };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_abs_ocl<float>(const int N, const float* X, float* Y);
template void caffe_gpu_abs_ocl<double>(const int N, const double* X,
    double* Y);

template <typename Dtype>
void caffe_gpu_div(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
  std::string kernel_name = "div" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &n);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) n };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_div<float>(const int n, const float* a, const float* b,
    float* y);
template void caffe_gpu_div<double>(const int n, const double* a,
    const double* b, double* y);

template <typename Dtype>
void caffe_gpu_add_scalar(const int n, const Dtype alpha, Dtype* top_data) {
  std::string kernel_name = "add_scalar" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &n);
  ret |= clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*) &alpha);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) n };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_add_scalar<float>(const int n, const float alpha,
    float* top_data);
template void caffe_gpu_add_scalar<double>(const int n, const double alpha,
    double* top_data);

template <typename Dtype>
void caffe_gpu_mul(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
  std::string kernel_name = "element_mul" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &n);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*) &b);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) n };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_mul<float>(const int n, const float* a, const float* b,
    float* y);
template void caffe_gpu_mul<double>(const int n, const double* a,
    const double* b, double* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype alpha, Dtype* y) {
  std::string kernel_name = "powx" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &n);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &a);
  ret |= clSetKernelArg(Kernel, 2, sizeof(Dtype), (void*) &alpha);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*) &y);
  OCL_CHECK(ret);
  size_t Global_Work_Size[] = { (size_t) n };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_powx<float>(const int n, const float* a,
    const float alpha, float* y);
template void caffe_gpu_powx<double>(const int n, const double* a,
    const double alpha, double* y);

template <typename Dtype>
void DropoutForward(const int count, const Dtype* bottom_data,
    const  unsigned int* MaskMem, const unsigned int threshold,
    const float scale_, Dtype* top_data) {
	std::string kernel_name = "DropoutForward" + get_dtype_suffix<Dtype>();
	cl_kernel kernel = amdDevice.GetKernel(kernel_name);

	cl_int ret;
	ret = clSetKernelArg(kernel,  0, sizeof(cl_int), (void*) &count);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &MaskMem);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*) &threshold);
        ret |= clSetKernelArg(kernel, 4, sizeof(cl_float), (void*) &scale_);
	ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &top_data);
	OCL_CHECK(ret);

	size_t Global_Work_Size[] = { (size_t) count };
	size_t Local_Work_Size[] = { 256 };
	OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
					Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void DropoutForward<float>(const int count, const float* bottom_data,
		const unsigned int* MaskMem, const unsigned int threshold, 
                const float scale_, float* top_data);
template void DropoutForward<double>(const int count, const double* bottom_data,
		const unsigned int* MaskMem, const unsigned int threshold, 
                const float scale_, double* top_data);

template <typename Dtype>
void DropoutBackward(const int count, const Dtype* top_diff, const unsigned int* MaskMem,
		const unsigned int threshold_, const float scale_, Dtype* bottom_diff) {
	std::string kernel_name = "DropoutBackward" + get_dtype_suffix<Dtype>();
	cl_kernel kernel = amdDevice.GetKernel(kernel_name);

	cl_int ret;
	ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &count);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &top_diff);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &MaskMem);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*) &threshold_);
	ret |= clSetKernelArg(kernel, 4, sizeof(cl_float), (void*) &scale_);
	ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &bottom_diff);
	OCL_CHECK(ret);

	size_t Global_Work_Size[] = { (size_t) count };
	size_t Local_Work_Size[] = { 256 };
	OCL_CHECK(
			clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
					Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void DropoutBackward<float>(const int count, const float* top_diff,
		const unsigned int* MaskMem, const unsigned int threshold_, const float scale_,
		float* bottom_diff);
template void DropoutBackward<double>(const int count, const double* top_diff,
		const unsigned int* MaskMem, const unsigned int  threshold_, const float scale_,
		double* bottom_diff);

template <typename Dtype>
void BNLLForward(const int count, const Dtype* bottom_data, Dtype *top_data) {
  std::string kernel_name = "BNLLForward" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &top_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void BNLLForward<float>(const int count, const float* bottom_data,
    float *top_data);
template void BNLLForward<double>(const int count, const double* bottom_data,
    double *top_data);

template <typename Dtype>
void BNLLBackward(const int count, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype *bottom_diff) {
  std::string kernel_name = "BNLLBackward" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bottom_data);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void BNLLBackward<float>(const int count, const float* top_diff,
    const float* bottom_data, float *bottom_diff);
template void BNLLBackward<double>(const int count, const double* top_diff,
    const double* bottom_data, double *bottom_diff);

template <typename Dtype>
void Concat(const int nthreads, const Dtype* in_data, const bool forward,
    const int num_concats, const int concat_size, const int top_concat_axis,
    const int bottom_concat_axis, const int offset_concat_axis,
    Dtype *out_data) {
  std::string kernel_name = "Concat" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);
  int k_forward = (forward == true) ? 1 : 0;
  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &in_data);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &k_forward);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &num_concats);
  ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &concat_size);
  ret |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*) &top_concat_axis);
  ret |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*) &bottom_concat_axis);
  ret |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*) &offset_concat_axis);
  ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*) &out_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) nthreads };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void Concat<float>(const int nthreads, const float* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, float *out_data);
template void Concat<double>(const int nthreads, const double* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, double *out_data);

template <typename Dtype>
void CLLBackward(const int count, const int channels, const Dtype margin,
    const bool legacy_version, const Dtype alpha, const Dtype* y,
    const Dtype* diff, const Dtype* dist_sq, Dtype *bottom_diff) {
  std::string kernel_name = "CLLBackward" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &count);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(kernel, 2, sizeof(Dtype), (void*) &margin);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_bool), (void*) &legacy_version);
  ret |= clSetKernelArg(kernel, 4, sizeof(Dtype), (void*) &alpha);
  ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &y);
  ret |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*) &diff);
  ret |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*) &dist_sq);
  ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void CLLBackward<float>(const int count, const int channels,
    const float margin, const bool legacy_version, const float alpha,
    const float* y, const float* diff, const float* dist_sq,
    float *bottom_diff);
template void CLLBackward<double>(const int count, const int channels,
    const double margin, const bool legacy_version, const double alpha,
    const double* y, const double* diff, const double* dist_sq,
    double *bottom_diff);

template <typename Dtype>
void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data,
    int* mask) {
  std::string kernel_name = "MaxForward" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &bottom_data_a);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &bottom_data_b);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &blob_idx);
  ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &top_data);
  ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &mask);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) nthreads };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void MaxForward<float>(const int nthreads, const float* bottom_data_a,
    const float* bottom_data_b, const int blob_idx, float* top_data, int* mask);
template void MaxForward<double>(const int nthreads,
    const double* bottom_data_a, const double* bottom_data_b,
    const int blob_idx, double* top_data, int* mask);

template <typename Dtype>
void MaxBackward(const int nthreads, const Dtype* top_diff, const int blob_idx,
    const int* mask, Dtype* bottom_diff) {
  std::string kernel_name = "MaxBackward" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);

  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &top_diff);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &blob_idx);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &mask);
  ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &bottom_diff);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) nthreads };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void MaxBackward<float>(const int nthreads, const float* top_diff,
    const int blob_idx, const int* mask, float* bottom_diff);
template void MaxBackward<double>(const int nthreads, const double* top_diff,
    const int blob_idx, const int* mask, double* bottom_diff);

template <typename Dtype>
void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data) {
  std::string kernel_name = "Slice" + get_dtype_suffix<Dtype>();
  cl_kernel kernel = amdDevice.GetKernel(kernel_name);
  int k_forward = (forward == true) ? 1 : 0;
  cl_int ret;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &nthreads);
  ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &in_data);
  ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &k_forward);
  ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &num_slices);
  ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &slice_size);
  ret |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*) &bottom_slice_axis);
  ret |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*) &top_slice_axis);
  ret |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*) &offset_slice_axis);
  ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*) &out_data);
  OCL_CHECK(ret);

  size_t Global_Work_Size[] = { (size_t) nthreads };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void Slice<float>(const int nthreads, const float* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, float* out_data);
template void Slice<double>(const int nthreads, const double* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, double* out_data);

template <typename Dtype>
void ocl_conv(Dtype* bottom_data, Dtype* top_data, Dtype* weights, Dtype* bias,
    int channel_in, int width, int height, int channel_out, int width_out,
    int height_out, int kernel_w, int kernel_h, int stride, int pad,
    int batch_sz) {
}
template void ocl_conv<float>(float* bottom_data, float* top_data,
    float* weights, float* bias, int channel_in, int width, int height,
    int channel_out, int width_out, int height_out, int kernel_w, int kernel_h,
    int stride, int pad, int batch_sz);
template void ocl_conv<double>(double* bottom_data, double* top_data,
    double* weights, double* bias, int channel_in, int width, int height,
    int channel_out, int width_out, int height_out, int kernel_w, int kernel_h,
    int stride, int pad, int batch_sz);

#endif

}  // namespace caffe

