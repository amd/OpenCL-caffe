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

template <class Dtype>
__kernel void MaxForward(const int nthreads, __global const Dtype* bottom_data_a,
    __global const Dtype* bottom_data_b, const int blob_idx, __global Dtype* top_data,
    __global int* mask) {
  int index = get_global_id(0);
  if(index < nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}
template __attribute__((mangled_name(MaxForward_float))) __kernel void MaxForward(const int nthreads, __global const float* bottom_data_a,
    __global const float* bottom_data_b, const int blob_idx, __global float* top_data,
    __global int* mask);
template __attribute__((mangled_name(MaxForward_double))) __kernel void MaxForward(const int nthreads, __global const double* bottom_data_a,
    __global const double* bottom_data_b, const int blob_idx, __global double* top_data,
    __global int* mask);

template <class Dtype>
__kernel void MaxBackward(const int nthreads, __global const Dtype* top_diff,
    const int blob_idx, __global const int* mask, __global Dtype* bottom_diff) {
  int index = get_global_id(0);
  if(index < nthreads) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}
template __attribute__((mangled_name(MaxBackward_float))) __kernel void MaxBackward(const int nthreads, __global const float* top_diff,
    const int blob_idx, __global const int* mask, __global float* bottom_diff);
template __attribute__((mangled_name(MaxBackward_double))) __kernel void MaxBackward(const int nthreads, __global const double* top_diff,
    const int blob_idx, __global const int* mask, __global double* bottom_diff);
