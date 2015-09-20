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

#ifndef _CAFFE_UTIL_OCL_WRAPPER_HPP_
#define _CAFFE_UTIL_OCL_WRAPPER_HPP_

namespace caffe {

typedef unsigned int uint32_t;

template <typename dtype> inline std::string get_dtype_suffix() {
  dtype x;
  const char type = typeid(x).name()[0];
  std::string suffix;
  switch (type) {
  case 'i':
    suffix = "_int";
    break;
  case 'd':
    suffix = "_double";
    break;
  case 'f':
  default:
    suffix = "_float";
  }
  return suffix;
}

#ifndef CPU_ONLY
template <typename Dtype>
void transform_gpu(Dtype* src, Dtype* dst, const int top_offset, const int N_,
    const int M_, const int packing_num);

template <typename Dtype>
void opttrans(const Dtype* data_im, const int im_offset, const int channels,
    const int height, const int width, Dtype* data_opt, const int opt_offset,
    const int optnum);

template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* bottom_data, Dtype* scale_data);

template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* scale, Dtype* data);

template <typename Dtype>
Dtype softmax_gpu(cl_kernel Kernel, const int num, const int dim,
    const Dtype* prob_data, const Dtype* label, cl_mem d_loss);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, const int dim, Dtype* data,
    const Dtype* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    Dtype* top_data);

template <typename Dtype>
void MaxPoolForward(const int count, const Dtype* bottom_data, const int clnum,
    const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, Dtype* top_data, int* mask,
    Dtype* top_mask);

template <typename Dtype>
void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff);

template <typename Dtype>
void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff);

template <typename Dtype>
void StoPoolBackward(const int nthreads, const Dtype* const rand_idx,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, Dtype* const bottom_diff);
template <typename Dtype>
void SigmoidForward(const int count, const Dtype* bottom_data, Dtype* top_data);

template <typename Dtype>
void SigmoidBackward(const int count, const Dtype* top_diff,
    const Dtype* top_data, Dtype* bottom_diff);

template <typename Dtype>
void TanHForward(const int count, const Dtype* bottom_data, Dtype* top_data);

template <typename Dtype>
void TanHBackward(const int count, const Dtype* top_diff, const Dtype* top_data,
    Dtype* bottom_diff);

template <typename Dtype>
void ThresholdForward(const int count, const Dtype threshold,
    const Dtype* bottom_data, Dtype* top_data);

template <typename Dtype>
void ave_pool_fp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const int clnum, const int channels_,
    const int height_, const int width_, const int pooled_height_,
    const int pooled_width_, const int kernel_size_, const int stride_,
    const int pad_, Dtype* top_data);

template <typename Dtype>
void AvePoolForward(const int count, const Dtype* bottom_data, const int clnum,
    const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    const int pad_h_, const int pad_w_, Dtype* top_data);

template <typename Dtype>
void StoPoolForwardTrain(const int count, const Dtype* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    Dtype* idx_data, Dtype* top_data);

template <typename Dtype>
void StoPoolForwardTest(const int count, const Dtype* bottom_data,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_h_,
    const int kernel_w_, const int stride_h_, const int stride_w_,
    Dtype* top_data);

template <typename Dtype>
void max_pool_bp_gpu(cl_kernel Kernel, const int count,
    const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff,
    const int clnum, const int channels_, const int height_, const int width_,
    const int pooled_height_, const int pooled_width_, const int kernel_size_,
    const int stride_, Dtype* bottom_diff);

template <typename Dtype>
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff,
    const int clnum, const int channels_, const int intheight_,
    const int width_, const int pooled_height_, const int pooled_width_,
    const int kernel_size_, const int stride_, const int pad_,
    Dtype* bottom_diff);

template <typename Dtype>
void PReLUForward(const int count, const int channels, const int dim,
    const Dtype* bottom_data, Dtype* top_data, const Dtype* slope_data,
    const int div_factor);

template <typename Dtype>
void PReLUBackward(const int count, const int channels, const int dim,
    const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff,
    const Dtype* slope_data, const int div_factor);

template <typename Dtype>
void PReLUParamBackward(const int count, const Dtype* top_diff,
    const int offset_out, const Dtype* bottom_data, const int offset_in,
    Dtype* bottom_diff);

template <typename Dtype>
void ReLUForward(const int count, const Dtype* bottom_data, Dtype* top_data,
    Dtype negative_slope);

template <typename Dtype>
void ReLUBackward(const int count, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype* bottom_diff, Dtype negative_slope);

template <typename Dtype>
void caffe_gpu_div(const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void DropoutForward(const int count, const Dtype* bottom_data,
		const unsigned int* MaskMem, const unsigned int threshold, const float scale_, Dtype *top_data);

template <typename Dtype>
void DropoutBackward(const int count, const Dtype* top_diff, const unsigned int* MaskMem,
		const unsigned int threshold_, const float scale_, Dtype* bottom_diff);

template <typename Dtype>
void caffe_gpu_bernoulli(int* a, const unsigned int n, Dtype inf, Dtype sup,
    Dtype threshold);

void caffe_gpu_uniform(const unsigned int n, unsigned int *r, unsigned int _seed = 0);

template <typename Dtype>
void caffe_gpu_uniform(Dtype* a, const unsigned int n, Dtype inf, Dtype sup, unsigned int  _seed = 0);

template <typename Dtype>
void caffe_gpu_gaussian(Dtype* a, const unsigned int n, Dtype E, Dtype V);

template <typename Dtype>
void caffe_gpu_abs_ocl(const int N, const Dtype* X, Dtype * Y);

template <typename Dtype>
void caffe_gpu_signbit(const int N, const Dtype* X, Dtype * Y);

template <typename Dtype>
void caffe_gpu_sign_ocl(const int N, const Dtype* X, Dtype * Y);

template <typename Dtype>
void caffe_gpu_sign_with_offset_ocl(const int N, const Dtype* X, const int offx,  Dtype * Y, const int offy);

template <typename Dtype>
void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out);

template <typename Dtype>
void kernel_channel_subtract(const int count, const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data);

template <typename Dtype>
void kernel_powx(const int count, const Dtype* data, const Dtype alpha,
    Dtype* out);

template <typename Dtype>
void kernel_div(const int count, const Dtype* a, const Dtype* b, Dtype* out);

template <typename Dtype>
void kernel_add(const int count, const Dtype* a, const Dtype* b, Dtype* out);

template <typename Dtype>
void kernel_mul(const int count, const Dtype* a, const Dtype* b, Dtype* out);

template <typename Dtype>
void kernel_log(const int count, const Dtype* data, Dtype* out);

template <typename Dtype>
void kernel_sub(const int count, const Dtype* a, const Dtype* b, Dtype* out);

template <typename Dtype>
void kernel_add_scalar(const int count, const Dtype data, Dtype* out);

template <typename Dtype>
void kernel_exp(const int count, const Dtype* data, Dtype* out);

template <typename Dtype>
void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum);

template <typename Dtype>
void kernel_channel_div(const int count, const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data);

template <typename Dtype>
void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot);

template <typename Dtype>
void SoftmaxLossForwardGPU(const int nthreads, const Dtype* prob_data,
    const Dtype* label, Dtype* loss, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, Dtype* counts);

template <typename Dtype>
void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
    const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
    const int spatial_dim, const bool has_ignore_label_,
    const int ignore_label_, Dtype* counts);

template <typename Dtype>
void caffe_gpu_add(const int n, const Dtype* in1, const Dtype* in2, Dtype* y);

template <typename Dtype>
void caffe_gpu_add_scalar(const int n, const Dtype alpha, Dtype* top_data);

template <typename Dtype>
void LRNFillScale(const int nthreads, const Dtype* const in, const int num,
    const int channels, const int height, const int width, const int size,
    const Dtype alpha_over_size, const Dtype k, Dtype* const scale);

template <typename Dtype>
void LRNComputeOutput(int nthreads, const Dtype* in, Dtype* scale,
    Dtype negative_beta, Dtype* out);

template <typename Dtype>
void LRNComputeDiff(const int nthreads, const Dtype* const bottom_data,
    const Dtype* const top_data, const Dtype* const scale,
    const Dtype* const top_diff, const int num, const int channels,
    const int height, const int width, const int size,
    const Dtype negative_beta, const Dtype cache_ratio,
    Dtype* const bottom_diff);
template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype alpha, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void BNLLForward(const int count, const Dtype* bottom_data, Dtype *top_data);

template <typename Dtype>
void BNLLBackward(const int count, const Dtype* top_diff,
    const Dtype* bottom_data, Dtype *bottom_diff);

template <typename Dtype>
void Concat(const int nthreads, const Dtype* in_data, const bool forward,
    const int num_concats, const int concat_size, const int top_concat_axis,
    const int bottom_concat_axis, const int offset_concat_axis,
    Dtype *out_data);

template <typename Dtype>
void CLLBackward(const int count, const int channels, const Dtype margin,
    const bool legacy_version, const Dtype alpha, const Dtype* y,
    const Dtype* diff, const Dtype* dist_sq, Dtype *bottom_diff);

template <typename Dtype>
void MaxForward(const int nthreads, const Dtype* bottom_data_a,
    const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data, int* mask);

template <typename Dtype>
void MaxBackward(const int nthreads, const Dtype* top_diff, const int blob_idx,
    const int* mask, Dtype* bottom_diff);

template <typename Dtype>
void Slice(const int nthreads, const Dtype* in_data,
    const bool forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, Dtype* out_data);
#endif
}
#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
// namespace caffe
