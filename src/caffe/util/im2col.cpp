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

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename dtype> extern std::string get_dtype_suffix();

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] = data_im[(c_im
              * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels, const int height,
    const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] += data_col[(c
              * height_col + h) * width_col + w];
      }
    }
  }
}

template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_im);

#ifndef CPU_ONLY
template <typename Dtype>
void col2im_gpu_opt(const Dtype* data_col, const int col_offset,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_im, const int img_offset,
    int optnum) {
  std::string kernel_name = "col2im_opt" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height * width * optnum;

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num_kernels);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data_col);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &col_offset);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &kernel_w);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &height_col);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &width_col);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &data_im);
  ret |= clSetKernelArg(Kernel, 15, sizeof(cl_int), (void*) &img_offset);
  ret |= clSetKernelArg(Kernel, 16, sizeof(cl_int), (void*) &optnum);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) num_kernels };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void col2im_gpu_opt<float>(const float* data_col, const int col_offset,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_im, const int img_offset,
    int optnum);
template void col2im_gpu_opt<double>(const double* data_col,
    const int col_offset, const int channels, const int height, const int width,
   const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,   double* data_im,
    const int img_offset, int optnum);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_col, const int col_offset) {
  std::string kernel_name = "im2col" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num_kernels);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data_im);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &img_offset);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &kernel_w);

  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &height_col);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &width_col);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_mem), (void*) &data_col);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_int), (void*) &col_offset);

  size_t uiGlobal_Work_Size[] = { (size_t) num_kernels };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));

}

template void im2col_gpu<float>(const float* data_im, const int img_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col, const int col_offset);
template void im2col_gpu<double>(const double* data_im, const int img_offset,
    const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col, const int col_offset);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int col_offset, const int channels, const int height,
    const int width,  const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_im, const int img_offset) {
  std::string kernel_name = "col2im" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num_kernels);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data_col);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &col_offset);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &patch_h);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &patch_w);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &height_col);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &width_col);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &data_im);
  ret |= clSetKernelArg(Kernel, 15, sizeof(cl_int), (void*) &img_offset);

  size_t uiGlobal_Work_Size[] = { (size_t) num_kernels };
  size_t uiLocal_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void col2im_gpu<float>(const float* data_col, const int col_offset,
    const int channels, const int height, const int width, const int patch_h,
    const int patch_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im, const int img_offset);
template void col2im_gpu<double>(const double* data_col, const int col_offset,
    const int channels, const int height, const int width, const int patch_h,
    const int patch_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im, const int img_offset);

template <typename Dtype>
void im2col_gpu_opt(const Dtype* data_im, const int img_offset,
    const int channels, const int height, const int width,const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_col, const int col_offset,
    int optnum) {

  std::string kernel_name = "im2col_opt" + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = optnum * channels * height_col * width_col;

  cl_int ret;
  ret = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*) &num_kernels);
  ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*) &data_im);
  ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &channels);
  ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &img_offset);
  ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*) &height);
  ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*) &width);
  ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*) &kernel_h);
  ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*) &kernel_w);
  ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*) &pad_h);
  ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*) &pad_w);
  ret |= clSetKernelArg(Kernel, 10, sizeof(cl_int), (void*) &stride_h);
  ret |= clSetKernelArg(Kernel, 11, sizeof(cl_int), (void*) &stride_w);
  ret |= clSetKernelArg(Kernel, 12, sizeof(cl_int), (void*) &height_col);
  ret |= clSetKernelArg(Kernel, 13, sizeof(cl_int), (void*) &width_col);
  ret |= clSetKernelArg(Kernel, 14, sizeof(cl_mem), (void*) &data_col);
  ret |= clSetKernelArg(Kernel, 15, sizeof(cl_int), (void*) &col_offset);
  ret |= clSetKernelArg(Kernel, 16, sizeof(cl_int), (void*) &optnum);
  OCL_CHECK(ret);

  size_t uiGlobal_Work_Size[] = { (size_t) num_kernels };
  size_t uiLocal_Work_Size[] = { (size_t)(256 - 256 % width_col) };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void im2col_gpu_opt<float>(const float* data_im, const int img_offset,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, float* data_col, const int col_offset,
    int optnum);
template void im2col_gpu_opt<double>(const double* data_im,
    const int img_offset, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, double* data_col,
    const int col_offset, int optnum);

#endif
}  // namespace caffe
