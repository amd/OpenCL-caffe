// Copyright 2014 AMD DNN contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include "caffe/common.hpp"
#include "caffe/util/ocl_util.hpp"
namespace caffe {


template <typename Dtype>
void ocl_memset(cl_kernel Kernel, Dtype* buffer, const Dtype value, const int count){
    cl_int err=0;
    //cl_kernel Kernel = clCreateKernel(amdDevice.Program, "oclmemfloat", &err);
    //if(NULL==Kernel){
    //    fprintf(stderr, "Failed to create kernel %d\n", err);
    //}   
 
    err=clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*)&buffer);
    err|=clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*)&value);
    err|=clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&count);
    OCL_CHECK(err);
 
    size_t Global_Work_Size[1] = {count};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));

}

// Explicit instantiation
template void ocl_memset<float>(cl_kernel Kernel, float* buffer, const float value, const int count);
template void ocl_memset<double>(cl_kernel Kernel, double* buffer, const double value, const int count);


void ocl_memset(cl_kernel Kernel, cl_mem buffer, const int value, const int count){
   cl_int err=0;
  //  cl_kernel Kernel = clCreateKernel(amdDevice.Program, "OCL_memset2", &err);
   // if(NULL==Kernel){
   //     fprintf(stderr, "Failed to create kernel %d\n", err);
   // }

    err =clSetKernelArg(Kernel, 0, sizeof(cl_mem), (void*)&buffer);
    err|=clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&value);
    err|=clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&count);
    OCL_CHECK(err);

    size_t Global_Work_Size[] = {count};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));

}

void eventCallback(cl_event event, cl_int event_status, void* user_data){
    printf("The calling\n");
    int err = 0;
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    double run_time;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &ev_start_time, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);
    run_time = (double)(ev_end_time - ev_start_time);
    printf("The kernel's running time is %f s\n", run_time * 1.0e-9);
}


}  // namespace caffe
