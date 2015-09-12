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
__kernel void SoftmaxLossForwardGPU(const int nthreads,
    __global T* prob_data, __global T* label,__global T* loss,
    int num, int dim, int spatial_dim,
    bool has_ignore_label_, int ignore_label_,
    __global T* counts) {
  int index = get_global_id(0);
  if(index < nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
              T(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template __attribute__ ((mangled_name(SoftmaxLossForwardGPU_float))) __kernel void SoftmaxLossForwardGPU(int nthreads,
    __global float* prob_data, __global float* label,__global float* loss,
    int num, int dim, int spatial_dim,
    bool has_ignore_label_, int ignore_label_,
    __global float* counts);
template __attribute__ ((mangled_name(SoftmaxLossForwardGPU_double))) __kernel void SoftmaxLossForwardGPU(int nthreads,
    __global double* prob_data, __global double* label,__global double* loss,
    int num, int dim, int spatial_dim,
    bool has_ignore_label_, int ignore_label_,
    __global double* counts);

template <class T>
__kernel void SoftmaxLossBackwardGPU(int nthreads, __global T* top,
    __global T* label,__global T* bottom_diff, int num, int dim,
    int spatial_dim, bool has_ignore_label_,
    int ignore_label_, T* counts) {
  const int channels = dim / spatial_dim;
  int index = get_global_id(0);
  if(index < nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
template __attribute__ ((mangled_name(SoftmaxLossBackwardGPU_float))) __kernel void SoftmaxLossBackwardGPU(int nthreads, __global float* top,
    __global float* label,__global float* bottom_diff, int num, int dim,
    int spatial_dim, bool has_ignore_label_,
    int ignore_label_, float* counts);

template __attribute__ ((mangled_name(SoftmaxLossBackwardGPU_double))) __kernel void SoftmaxLossBackwardGPU(int nthreads, __global double* top,
    __global double* label,__global double* bottom_diff, int num, int dim,
    int spatial_dim, bool has_ignore_label_,
    int ignore_label_, double* counts);

template <class T>
__kernel void scal (const int num, const T alpha, __global T* data) {
  int index = get_global_id(0);
  int total = get_global_size(0);
  for(index; index < num; index += total) {
    data[index] = data[index] * alpha;
  }
}

template __attribute__ ((mangled_name(scal_float))) __kernel void scal (const int num, const float alpha, __global float* data);
template __attribute__ ((mangled_name(scal_double))) __kernel void scal (const int num, const double alpha, __global double* data);
