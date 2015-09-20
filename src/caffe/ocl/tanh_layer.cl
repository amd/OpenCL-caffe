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
__kernel void TanHForward(const int count, __global T* in, __global T* out) {
  int index = get_global_id(0);
  if(index < count)
  out[index] =tanh(in[index]);
}

template __attribute__ ((mangled_name(TanHForward_float))) __kernel void TanHForward(const int count, __global float* in, __global float* out);
template __attribute__ ((mangled_name(TanHForward_double))) __kernel void TanHForward(const int count, __global double* in, __global double* out);

template <class T>
__kernel void TanHBackward(const int count, __global T* in_diff, __global T* out_data,__global T* out_diff) {
  int index = get_global_id(0);
  const T tanhx = out_data[index];
  if(index < count)
  out_diff[index] = in_diff[index] * ( 1- tanhx * tanhx);
}

template __attribute__ ((mangled_name(TanHBackward_float))) __kernel void TanHBackward(const int count, __global float* in_diff, __global float* out_data, __global float* out_diff);
template __attribute__ ((mangled_name(TanHBackward_double))) __kernel void TanHBackward(const int count, __global double* in_diff, __global double* out_data, __global double* out_diff);
