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

template <class T>
__kernel void LRNComputeOutput(const int nthreads, __global T* in, __global T* scale, const T negative_beta, __global T* out) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index += tmp)
  out[index] = in[index] * pow(scale[index], negative_beta);
}
template __attribute__((mangled_name(LRNComputeOutput_float))) __kernel void LRNComputeOutput(const int nthreads, __global float* in, __global float* scale, const float negative_beta, __global float* out);
template __attribute__((mangled_name(LRNComputeOutput_double))) __kernel void LRNComputeOutput(const int nthreads, __global double* in, __global double* scale, const double negative_beta, __global double* out);

template <class T>
__kernel void LRNFillScale(const int nthreads, __global T* in, const int num, const int channels, const int height, const int width, const int size, const T alpha_over_size, const T k, __global T* scale) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index += tmp) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    in = in + offset;
    scale = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    T accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step]
        * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step]
        * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}
template __attribute__((mangled_name(LRNFillScale_float))) __kernel void LRNFillScale (const int nthreads, __global float* in, const int num, const int channels, const int height, const int width, const int size, const float alpha_over_size, const float k, __global float* scale);
template __attribute__((mangled_name(LRNFillScale_double))) __kernel void LRNFillScale (const int nthreads, __global double* in, const int num, const int channels, const int height, const int width, const int size, const double alpha_over_size, const double k, __global double* scale);

template <class T>
__kernel void LRNComputeDiff(const int nthreads, __global T* bottom_data, __global T* top_data, __global T* scale, __global T* top_diff, const int num, const int channels, const int height, const int width, const int size, const T negative_beta, const T cache_ratio, __global T* bottom_diff) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index += tmp) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    T accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
      scale[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
      scale[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
        top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] =
      top_diff[(head - post_pad) * step]
      * pow(scale[(head - post_pad) * step], negative_beta)
      - cache_ratio * bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
        top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] =
      top_diff[(head - post_pad) * step]
      * pow(scale[(head - post_pad) * step], negative_beta)
      - cache_ratio * bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

template __attribute__((mangled_name(LRNComputeDiff_float))) __kernel void LRNComputeDiff(const int nthreads, __global float* bottom_data, __global float* top_data, __global float* scale, __global float* top_diff, const int num, const int channels, const int height, const int width, const int size, const float negative_beta, const float cache_ratio, __global float* bottom_diff);
template __attribute__((mangled_name(LRNComputeDiff_double))) __kernel void LRNComputeDiff(const int nthreads, __global double* bottom_data, __global double* top_data, __global double* scale, __global double* top_diff, const int num, const int channels, const int height, const int width, const int size, const double negative_beta, const double cache_ratio, __global double* bottom_diff);
