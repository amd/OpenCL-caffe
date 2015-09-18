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

#pragma OPENCL EXTENSION cl_amd_printf : enable

template <class T>
__kernel void OCL_memset(__global T* buffer, const T value, const int size, const int buf_offset) {
  int gdx = get_global_id(0);
  buffer += buf_offset;
  if(gdx < size) {
    buffer[gdx] = value;
  }
}

template __attribute__((mangled_name(oclmem_int))) __kernel void OCL_memset(__global int* buffer, const int value, const int size, const int buf_offset);
template __attribute__((mangled_name(oclmem_float))) __kernel void OCL_memset(__global float* buffer, const float value, const int size, const int buf_offset);
template __attribute__((mangled_name(oclmem_double))) __kernel void OCL_memset(__global double* buffer, const double value, const int size, const int buf_offset);

__kernel void OCL_memset2(__global int* buffer, const int value, const int size) {
  int gdx = get_global_id(0);
  if(gdx < size) {
    buffer[gdx] = value;
  }
}

template <class T>
__kernel void caffe_gpu_sign(const int N, __global T* X, __global T* Y) {
  int gdx = get_global_id(0);
  if(gdx < N) {
    Y[gdx] =((X[gdx]>0.0)-(X[gdx]<0.0));
  }
}

template __attribute__((mangled_name(caffe_gpu_sign_float))) __kernel void caffe_gpu_sign(const int N, __global float* X, __global float* Y);
template __attribute__((mangled_name(caffe_gpu_sign_double))) __kernel void caffe_gpu_sign(const int N, __global double* X, __global double* Y);

template <class T>
__kernel void caffe_gpu_sgnbit(const int N, __global T* X, __global T* Y) {
  int gdx = get_global_id(0);
  if(gdx < N) {
    Y[gdx] =(X[gdx] < 0.0);
  }
}

template __attribute__((mangled_name(caffe_gpu_sgnbit_float))) __kernel void caffe_gpu_sgnbit(const int N, __global float* X, __global float* Y);
template __attribute__((mangled_name(caffe_gpu_sgnbit_double))) __kernel void caffe_gpu_sgnbit(const int N, __global double* X, __global double* Y);

template <class T>
__kernel void caffe_gpu_sign_with_offset(const int N, __global T* X, const int offx,  __global T* Y, const int offy) {
  X += offx;
  Y += offy;
  int gdx = get_global_id(0);
  if(gdx < N) {
    Y[gdx] =((X[gdx]>0.0)-(X[gdx]<0.0));
  }
}
template __attribute__((mangled_name(caffe_gpu_sign_with_offset_float))) __kernel void caffe_gpu_sign_with_offset(const int N, __global float* X, const int offx,  __global float* Y, const int offy);
template __attribute__((mangled_name(caffe_gpu_sign_with_offset_double))) __kernel void caffe_gpu_sign_with_offset(const int N, __global double* X, const int offx,  __global double* Y, const int offy);

template <class T>
__kernel void caffe_gpu_abs(const int n, __global T* a, __global T* y) {
  int index = get_global_id(0);
  if(index < n) {
    y[index] = fabs(a[index]);
  }
}
template __attribute__((mangled_name(caffe_gpu_abs_float))) __kernel void caffe_gpu_abs(const int n, __global float* a, __global float* Y);
template __attribute__((mangled_name(caffe_gpu_abs_double))) __kernel void caffe_gpu_abs(const int n, __global double* a, __global double* Y);

template <class T>
__kernel void get_max(const int num, const int dim, __global T* data, __global T* out) {
  int index = get_global_id(0);
  if (index < num) {
    T maxval = -FLT_MAX;
    for (int i = 0; i < dim; i++)
    maxval = max( data[index*dim + i], maxval );
    out[index] = maxval;
  }
}

template __attribute__ ((mangled_name(get_max_float))) __kernel void get_max(const int num, const int dim, __global float* data, __global float* out);
template __attribute__ ((mangled_name(get_max_double))) __kernel void get_max(const int num, const int dim, __global double* data, __global double* out);

template <class T>
__kernel void exp (const int num, __global T* data, __global T* out) {
  int index = get_global_id(0);
  if (index < num)
  out[index] = exp(data[index]);
}

template __attribute__ ((mangled_name(exp_float))) __kernel void exp (const int num, __global float* data, __global float* out);
template __attribute__ ((mangled_name(exp_double))) __kernel void exp (const int num, __global double* data, __global double* out);

template <class T>
__kernel void kernel_sub(const int count, __global const T* a, __global const T* b, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = a[index] - b[index];
  }
}

template __attribute__ ((mangled_name(kernel_sub_float))) __kernel void kernel_sub(const int count, __global const float* a, __global const float* b, __global float* out);
template __attribute__ ((mangled_name(kernel_sub_double))) __kernel void kernel_sub(const int count, __global const double* a, __global const double* b, __global double* out);

template <class T>
__kernel void kernel_add(const int count, __global const T* a, __global const T* b, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = a[index] + b[index];
  }
}

template __attribute__ ((mangled_name(kernel_add_float))) __kernel void kernel_add(const int count, __global const float* a, __global const float* b, __global float* out);
template __attribute__ ((mangled_name(kernel_add_double))) __kernel void kernel_add(const int count, __global const double* a, __global const double* b, __global double* out);

template <class T>
__kernel void kernel_div(const int count, __global const T* a, __global const T* b, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = a[index] / b[index];
  }
}

template __attribute__ ((mangled_name(kernel_div_float))) __kernel void kernel_div(const int count, __global const float* a, __global const float* b, __global float* out);
template __attribute__ ((mangled_name(kernel_div_double))) __kernel void kernel_div(const int count, __global const double* a, __global const double* b, __global double* out);

template <class T>
__kernel void kernel_mul(const int count, __global const T* a, __global const T* b, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = a[index] * b[index];
  }
}

template __attribute__ ((mangled_name(kernel_mul_float))) __kernel void kernel_mul(const int count, __global const float* a, __global const float* b, __global float* out);
template __attribute__ ((mangled_name(kernel_mul_double))) __kernel void kernel_mul(const int count, __global const double* a, __global const double* b, __global double* out);

template <class T>
__kernel void kernel_powx(const int count, __global const T* data, const T alpha, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = pow(data[index], alpha);
  }
}

template __attribute__ ((mangled_name(kernel_powx_float))) __kernel void kernel_powx(const int count, __global const float* data, const float alpha, __global float* out);
template __attribute__ ((mangled_name(kernel_powx_double))) __kernel void kernel_powx(const int count, __global const double* data, const double alpha, __global double* out);

template <class T>
__kernel void kernel_exp(const int count, __global const T* data, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = exp(data[index]);
  }
}

template __attribute__ ((mangled_name(kernel_exp_float))) __kernel void kernel_exp(const int count, __global const float* data, __global float* out);
template __attribute__ ((mangled_name(kernel_exp_double))) __kernel void kernel_exp(const int count, __global const double* data, __global double* out);

template <class T>
__kernel void kernel_add_scalar(const int count, const T data, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = out[index] + data;
  }
}

template __attribute__ ((mangled_name(kernel_add_scalar_float))) __kernel void kernel_add_scalar(const int count, const float data, __global float* out);
template __attribute__ ((mangled_name(kernel_add_scalar_double))) __kernel void kernel_add_scalar(const int count, const double data, __global double* out);

template <class T>
__kernel void kernel_log(const int count, __global const T* data, __global T* out) {
  int index = get_global_id(0);
  if(index < count) {
    out[index] = log(data[index]);
  }
}

template __attribute__ ((mangled_name(kernel_log_float))) __kernel void kernel_log(const int count, __global const float* data, __global float* out);
template __attribute__ ((mangled_name(kernel_log_double))) __kernel void kernel_log(const int count, __global const double* data, __global double* out);

template <class T>
__kernel void diff (const int num, const int dim, __global T* data, __global T* label) {
  int index = get_global_id(0);
  int total = get_global_size(0);
  int offset;
  for(index; index < num; index += total) {
    offset = (int) label[index];
    data[index * dim + offset] -= 1;
  }
}

template __attribute__ ((mangled_name(diff_float))) __kernel void diff (const int num, const int dim, __global float* data, __global float* label);
template __attribute__ ((mangled_name(diff_double))) __kernel void diff (const int num, const int dim, __global double* data, __global double* label);

template <class T>
__kernel void div (const int n, __global const T* a, __global const T* b, __global T* y) {
  int index = get_global_id(0);
  if (index < n)
  y[index] = a[index] / b[index];
}

template __attribute__ ((mangled_name(div_float))) __kernel void div (const int n, __global const float* a, __global const float* b, __global float* y);
//template __attribute__ ((mangled_name(div_double))) __kernel void div (const int n, __global const double* a, __global const double* b, __global double* y);

template <class T>
__kernel void add_scalar (const int n, const T alpha, __global T* y) {
  int index = get_global_id(0);
  if (index < n)
  y[index] += alpha;
}

template __attribute__ ((mangled_name(add_scalar_float))) __kernel void add_scalar (const int n, const float alpha, __global float* y);
template __attribute__ ((mangled_name(add_scalar_double))) __kernel void add_scalar (const int n, const double alpha, __global double* y);

template <typename Dtype>
__kernel void caffe_gpu_add(const int n, const Dtype* in1, const Dtype* in2, Dtype* y) {
  int index = get_global_id(0);
  if (index < n)
  y[index] = in1[index] + in2[index];
}
template __attribute__ ((mangled_name(caffe_gpu_add_float))) __kernel void caffe_gpu_add(const int n, const float* in1, const float* in2, float* y);
template __attribute__ ((mangled_name(caffe_gpu_add_double))) __kernel void caffe_gpu_add(const int n, const double* in1, const double* in2, double* y);

template <class T>
__kernel void element_mul (const int n, __global const T* a, __global const T* b, __global T* y) {
  int index = get_global_id(0);
  if (index < n)
  y[index] = a[index] * b[index];
}

template __attribute__ ((mangled_name(element_mul_float))) __kernel void element_mul (const int n, __global const float* a, __global const float* b, __global float* y);
template __attribute__ ((mangled_name(element_mul_double))) __kernel void element_mul (const int n,__global const double* a, __global const double* b, __global double* y);

template <class T>
__kernel void powx (const int n, __global const T* a, const T alpha, __global T* y) {
  int index = get_global_id(0);
  if (index < n)
//           y[index] = a[index] + alpha;
  y[index] = pow(a[index], alpha);
}

template __attribute__ ((mangled_name(powx_float))) __kernel void powx (const int n, __global const float* a, const float alpha, __global float* y);
template __attribute__ ((mangled_name(powx_double))) __kernel void powx (const int n, __global const double* a, const double alpha, __global double* y);

