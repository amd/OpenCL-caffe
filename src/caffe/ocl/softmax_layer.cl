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
__kernel void softmax(__global T* prob_data, __global T* loss, __global T* label, int num, int dim, __local T* resultScratch) {

  int gid = get_global_id(0);
  int size = get_global_size(0);

  resultScratch[gid] = 0.0;
  for(int i = gid; i < num; i += size) {
    resultScratch[gid] += -log(prob_data[i * dim + static_cast<int>(label[i])]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(gid < 128)
  resultScratch[gid] += resultScratch[gid + 128];
  barrier(CLK_LOCAL_MEM_FENCE);
  if(gid < 64)
  resultScratch[gid] += resultScratch[gid + 64];
  if(gid < 32)
  resultScratch[gid] += resultScratch[gid + 32];
  if(gid < 16)
  resultScratch[gid] += resultScratch[gid + 16];
  if(gid < 8)
  resultScratch[gid] += resultScratch[gid + 8];
  if(gid < 4)
  resultScratch[gid] += resultScratch[gid + 4];
  if(gid < 2)
  resultScratch[gid] += resultScratch[gid + 2];
  if(gid < 1) {
    resultScratch[gid] += resultScratch[gid + 1];
    loss[0] = resultScratch[gid];
  }
}
template __attribute__ ((mangled_name(softmax_float))) __kernel void softmax (__global float* prob_data, __global float* loss, __global float* label, int num, int dim, __local float* resultScratch);
template __attribute__ ((mangled_name(softmax_double))) __kernel void softmax (__global double* prob_data, __global double* loss, __global double* label, int num, int dim, __local double* resultScratch);

template <class T>
__kernel void softmax_div (const int num, const int dim, __global T* scale, __global T* data) {
  //printf("softmax_div\n");
  int index = get_global_id(0);
  int total = get_global_size(0);
  for(index; index < num*dim; index += total) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template __attribute__ ((mangled_name(softmax_div_float))) __kernel void softmax_div (const int num, const int dim, __global float* scale, __global float* data);
template __attribute__ ((mangled_name(softmax_div_double))) __kernel void softmax_div (const int num, const int dim, __global double* scale, __global double* data);

template <class T>
__kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* out) {
  int index = get_global_id(0);
  if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template __attribute__ ((mangled_name(kernel_channel_max_float))) __kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const float* data, __global float* out);
template __attribute__ ((mangled_name(kernel_channel_max_double))) __kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const double* data, __global double* out);

template <class T>
__kernel void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_max, __global T* data) {
  int index = get_global_id(0);
  if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}
template __attribute__ ((mangled_name(kernel_channel_subtract_float))) __kernel void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, __global const float* channel_max, __global float* data);
template __attribute__ ((mangled_name(kernel_channel_subtract_double))) __kernel void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, __global const double* channel_max, __global double* data);

template <class T>
__kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* channel_sum) {
  int index = get_global_id(0);
  if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template __attribute__ ((mangled_name(kernel_channel_sum_float))) __kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const float* data, __global float* channel_sum);
template __attribute__ ((mangled_name(kernel_channel_sum_double))) __kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const double* data, __global double* channel_sum);

template <class T>
__kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_sum, __global T* data) {
  int index = get_global_id(0);
  if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template __attribute__ ((mangled_name(kernel_channel_div_float))) __kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const float* channel_sum, __global float* data);
template __attribute__ ((mangled_name(kernel_channel_div_double))) __kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const double* channel_sum, __global double* data);

template <class T>
__kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const T* data_1, __global const T* data_2,
    __global T* channel_dot) {
  int index = get_global_id(0);
  if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template __attribute__ ((mangled_name(kernel_channel_dot_float))) __kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const float* data_1, __global const float* data_2,
    __global float* channel_dot);
template __attribute__ ((mangled_name(kernel_channel_dot_double))) __kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const double* data_1, __global const double* data_2,
    __global double* channel_dot);
