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
__kernel void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
    __global const Dtype* y, __global const Dtype* diff, __global const Dtype* dist_sq,
    __global Dtype *bottom_diff) {
  int i = get_global_id(0);
  if(i < count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist(0.0);
      Dtype beta(0.0);
      if (legacy_version) {
        mdist = (margin - dist_sq[n]);
        beta = -alpha;
      } else {
        Dtype dist = sqrt(dist_sq[n]);
        mdist = (margin - dist);
        beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
      }
      if (mdist > 0.0) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template __attribute__((mangled_name(CLLBackward_float))) __kernel void CLLBackward(const int count, const int channels,
    const float margin, const bool legacy_version, const float alpha,
    __global const float* y, __global const float* diff, __global const float* dist_sq,
    __global float *bottom_diff);
template __attribute__((mangled_name(CLLBackward_double))) __kernel void CLLBackward(const int count, const int channels,
    const double margin, const bool legacy_version, const double alpha,
    __global const double* y, __global const double* diff, __global const double* dist_sq,
    __global double *bottom_diff);
