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

#define kBNLL_THRESHOLD  50.0

template <class T>
__kernel void BNLLForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n) {
    out[index] = in[index] > 0 ? in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));
  }
}
template __attribute__((mangled_name(BNLLForward_float))) __kernel void BNLLForward(const int n, __global const float* in, __global float* out);
template __attribute__((mangled_name(BNLLForward_double))) __kernel void BNLLForward(const int n, __global const double* in, __global double* out);

template <class T>
__kernel void BNLLBackward(const int n, __global const T* in_diff,
    __global const T* in_data, __global T* out_diff) {
  int index = get_global_id(0);
  if (index < n) {
    T expval = exp(min(in_data[index], T(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

template __attribute__((mangled_name(BNLLBackward_float))) __kernel void BNLLBackward(const int n, __global const float* in_diff,
    __global const float* in_data, __global float* out_diff);
template __attribute__((mangled_name(BNLLBackward_double))) __kernel void BNLLBackward(const int n, __global const double* in_diff,
    __global const double* in_data, __global double* out_diff);
