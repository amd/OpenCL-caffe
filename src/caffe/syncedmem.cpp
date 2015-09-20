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

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/ocl_util.hpp"

#define CL_MEM_USE_PERSISTENT_MEM_AMD (1 << 6)//specific for AMD devices

namespace caffe {

SyncedMemory::~SyncedMemory() {
#ifndef CPU_ONLY
  if (cpu_ptr_ && own_cpu_data_) {
    OCL_CHECK(
        clEnqueueUnmapMemObject(amdDevice.CommandQueue, (cl_mem) gpu_cache_ptr_,
            cpu_ptr_, 0, NULL, NULL));
    clFinish(amdDevice.CommandQueue);
  }
  if (gpu_cache_ptr_ && own_cpu_data_) {
    OCL_CHECK(clReleaseMemObject((cl_mem) gpu_cache_ptr_));
  }
  if (gpu_ptr_) {
    OCL_CHECK(clReleaseMemObject((cl_mem) gpu_ptr_));
  }

  clReleaseKernel (oclmem_kernel);
#endif
}

//begin: code written/modified by AMD.
#ifndef CPU_ONLY
void SyncedMemory::ocl_setup() {
  cl_int err = 0;
  oclmem_kernel = clCreateKernel(amdDevice.Program, "OCL_memset2", &err);
  OCL_CHECK(err);
}
#endif

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
#ifndef CPU_ONLY
    gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_ALLOC_HOST_PTR,
        size_, NULL, NULL);
    cpu_ptr_ = clEnqueueMapBuffer(amdDevice.CommandQueue,
        (cl_mem) gpu_cache_ptr_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, size_,
        0, NULL, NULL, NULL);
#else
    CaffeMallocHost(&cpu_ptr_, size_);
#endif
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU: {
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      gpu_cache_ptr_ = clCreateBuffer(amdDevice.Context, CL_MEM_ALLOC_HOST_PTR,
          size_, NULL, NULL);
      cpu_ptr_ = clEnqueueMapBuffer(amdDevice.CommandQueue,
          (cl_mem) gpu_cache_ptr_, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
          size_, 0, NULL, NULL, NULL);
      own_cpu_data_ = true;
    }
    OCL_CHECK(
        clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem) gpu_ptr_,
            (cl_mem) gpu_cache_ptr_, 0, 0, size_, 0, NULL, NULL));
    clFinish(amdDevice.CommandQueue);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  }
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED: {
    cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, size_,
        NULL, NULL);
    if (NULL == tmpMem) {
      fprintf(stderr, "Failed to create memory object\n");
      break;
    }
    ocl_memset(tmpMem, (int) 0, (int) (size_ / sizeof(int)));
    gpu_ptr_ = (void*) tmpMem;
    head_ = HEAD_AT_GPU;
    break;
  }
  case HEAD_AT_CPU: {
    if (gpu_ptr_ == NULL) {
      cl_mem tmpMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE,
          size_, NULL, NULL);
      if (NULL == tmpMem) {
        fprintf(stderr, "Failed to create memory object\n");
      }
      gpu_ptr_ = (void*) tmpMem;
    }
    OCL_CHECK(
        clEnqueueCopyBuffer(amdDevice.CommandQueue, (cl_mem) gpu_cache_ptr_,
            (cl_mem) gpu_ptr_, 0, 0, size_, 0, NULL, NULL));
    clFinish(amdDevice.CommandQueue);
    head_ = SYNCED;
    break;
  }
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*) cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost (cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*) gpu_ptr_;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
#endif
}

const void *SyncedMemory::gpu_cache_data() {
  return 0;
}

}  // namespace caffe

