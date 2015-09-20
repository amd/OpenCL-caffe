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

#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels, const int height,
    const int width, const int patch_h, const int patch_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w, Dtype* data_im);

#ifndef CPU_ONLY
template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int col_offset, const int height,
    const int width, const int channels, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_im, const int img_offset);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    Dtype* data_col, const int col_offset);

template <typename Dtype>
void im2col_gpu_opt(const Dtype* data_im, const int img_offset,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_col, const int col_offset,
    int optnum);

template <typename Dtype>
void col2im_gpu(cl_kernel Kernel, const Dtype* data_col, const int col_offset,
    const int channels, const int height, const int width, const int psize,
    const int pad, const int stride, Dtype* data_im, const int img_offset);

template <typename Dtype>
void col2im_gpu_opt(const Dtype* data_col, const int col_offset,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, Dtype* data_im, const int img_offset,
    int optnum);
#endif
}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
