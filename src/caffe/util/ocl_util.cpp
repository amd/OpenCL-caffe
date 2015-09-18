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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/util/ocl_util.hpp"
namespace caffe {

#ifndef CPU_ONLY

template <typename dtype> extern std::string get_dtype_suffix();

template <typename Dtype>
void ocl_memset(Dtype* buffer, const Dtype value, const int count, const int buf_offset) {
  std::string kernel_name = std::string("oclmem") + get_dtype_suffix<Dtype>();
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int err = 0;
  err = clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*) &buffer);
  err |= clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*) &value);
  err |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &count);
  err |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*) &buf_offset);
  OCL_CHECK(err);

  size_t Global_Work_Size[1] = { (size_t) count };
  size_t Local_Work_Size[1] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));

}

template void ocl_memset<int>(int* buffer, const int value, const int count, const int buf_offset);
template void ocl_memset<float>(float* buffer, const float value, const int count, const int buf_offset);
template void ocl_memset<double>(double* buffer, const double value, const int count, const int buf_offset);

void ocl_memset(cl_mem buffer, const int value,
    const int count) {
  std::string kernel_name = std::string("OCL_memset2");
  cl_kernel Kernel = amdDevice.GetKernel(kernel_name);
  cl_int err;
  err = clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*) &buffer);
  err |= clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*) &value);
  err |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*) &count);
  OCL_CHECK(err);

  size_t Global_Work_Size[] = { (size_t) count };
  size_t Local_Work_Size[] = { 256 };
  OCL_CHECK(
      clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL,
          Global_Work_Size, Local_Work_Size, 0, NULL, NULL));

}

void eventCallback(cl_event event, cl_int event_status, void* user_data) {
  cl_ulong ev_start_time = (cl_ulong) 0;
  cl_ulong ev_end_time = (cl_ulong) 0;
  double run_time;
  OCL_CHECK(
      clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
          sizeof(cl_ulong), &ev_start_time, NULL));
  OCL_CHECK(
      clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
          &ev_end_time, NULL));
  run_time = (double) (ev_end_time - ev_start_time);
  printf("The kernel's running time is %f s\n", run_time * 1.0e-9);
}

#endif
}  // namespace caffe
