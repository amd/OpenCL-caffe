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
__kernel void PReLUForward(const int count, const int channels, const int dim, __global T* in, __global T* out, __global T* slope_data, const int div_factor) {
  int index = get_global_id(0);
  if(index < count) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}
template __attribute__ ((mangled_name(PReLUForward_float))) __kernel void PReLUForward(const int count, const int channels, const int dim, __global float* in, __global float* out, __global float* slope_data, const int div_factor);
template __attribute__ ((mangled_name(PReLUForward_double))) __kernel void PReLUForward(const int count, const int channels, const int dim, __global double* in, __global double* out, __global double* slope_data, const int div_factor);

template <class T>
__kernel void PReLUBackward(const int count, const int channels, const int dim, __global T* in_diff, __global T* in_data, __global T* out_diff, __global T* slope_data, const int div_factor) {
  int index = get_global_id(0);
  if(index < count) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}
template __attribute__ ((mangled_name(PReLUBackward_float))) __kernel void PReLUBackward(const int count, const int channels, const int dim, __global float* in_diff, __global float* in_data, __global float* out_diff, __global float* slope_data, const int div_factor);
template __attribute__ ((mangled_name(PReLUBackward_double))) __kernel void PReLUBackward(const int count, const int channels, const int dim, __global double* in_diff, __global double* in_data, __global double* out_diff, __global double* slope_data, const int div_factor);

template <class T>
__kernel void PReLUParamBackward(const int count, __global T* in_diff, const int offset_in_diff, __global T* in_data, const int offset_in_data, __global T* out_diff) {
  int index = get_global_id(0);
  if(index < count) {
    in_diff += offset_in_diff;
    in_data += offset_in_data;
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
  }
}
template __attribute__ ((mangled_name(PReLUParamBackward_float))) __kernel void PReLUParamBackward(const int count, __global float* in_diff, const int offset_in_diff, __global float* in_data, const int offset_in_data, __global float* out_diff);
template __attribute__ ((mangled_name(PReLUParamBackward_double))) __kernel void PReLUParamBackward(const int count, __global double* in_diff, const int offset_in_diff, __global double* in_data, const int offset_in_data, __global double* out_diff);
