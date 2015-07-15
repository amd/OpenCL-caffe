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
void transform_gpu(cl_kernel Kernel, Dtype* src, Dtype* dst, const int top_offset, const int N_, const int M_, const int packing_num){
    cl_int ret;
    ret= clSetKernelArg(Kernel,0,sizeof(cl_mem),(void*)&src);
    OCL_CHECK(ret);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&dst);
    OCL_CHECK(ret);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&top_offset);
    OCL_CHECK(ret);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&N_);
    OCL_CHECK(ret);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&M_);
    OCL_CHECK(ret);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&packing_num);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size2[]={M_ * packing_num};
    size_t uiLocal_Work_Size2[]={256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size2, uiLocal_Work_Size2, 0, NULL, NULL) );
}

template void transform_gpu<float>(cl_kernel Kernel, float* src, float* dst, const int top_offset, const int N_, const int M_, const int packing_num);
template void transform_gpu<double>(cl_kernel Kernel, double* src, double* dst, const int top_offset, const int N_, const int M_, const int packing_num);

template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&bottom_data) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&scale_data) );
 
    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void get_max_gpu<float>(cl_kernel Kernel, const int num, const int dim, const float* bottom_data, float* scale_data);
template void get_max_gpu<double>(cl_kernel Kernel, const int num, const int dim, const double* bottom_data, double* scale_data);


template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&data) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&out) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void exp_gpu<float>(cl_kernel Kernel, const int num, const float* data, float* out);
template void exp_gpu<double>(cl_kernel Kernel, const int num, const double* data, double* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* scale, Dtype* data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&scale) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&data) );

    size_t Global_Work_Size[1] = {num*dim};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void softmax_div_gpu<float>(cl_kernel Kernel, const int num, const int dim, const float* scale, float* data);
template void softmax_div_gpu<double>(cl_kernel Kernel, const int num, const int dim, const double* scale, double* data);

template <typename Dtype>
Dtype softmax_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* prob_data, const Dtype* label, cl_mem d_loss){

    OCL_CHECK(clSetKernelArg(Kernel, 0, sizeof(cl_mem),     (void*)&prob_data));
    OCL_CHECK(clSetKernelArg(Kernel, 1, sizeof(cl_mem),  (void*)&d_loss));
    OCL_CHECK(clSetKernelArg(Kernel, 2, sizeof(cl_mem),   (void*)&label));
    OCL_CHECK(clSetKernelArg(Kernel, 3, sizeof(cl_int),   (void*)&num));
    OCL_CHECK(clSetKernelArg(Kernel, 4, sizeof(cl_int),   (void*)&dim));
    OCL_CHECK(clSetKernelArg(Kernel, 5, 256 * sizeof(Dtype),    NULL));

    size_t globalws[1] = {256};
    size_t localws[1] = {256};
    OCL_CHECK (clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, globalws, localws, 0, NULL, NULL) );
    void* h_loss = clEnqueueMapBuffer(amdDevice.CommandQueue, d_loss, CL_TRUE, CL_MAP_READ, 0, sizeof(Dtype), 0, NULL, NULL, NULL);
    Dtype loss = *(Dtype*)h_loss;
    clEnqueueUnmapMemObject(amdDevice.CommandQueue, d_loss, h_loss, 0, NULL, NULL);
    
    return loss;
}

// Explicit instantiation
template float softmax_gpu<float>(cl_kernel Kernel, const int num, const int dim, const float* prob_data, const float* label, cl_mem d_loss);
template double softmax_gpu<double>(cl_kernel Kernel, const int num, const int dim, const double* prob_data, const double* label, cl_mem d_loss);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*)&alpha) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&data) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void scal_gpu<float>(cl_kernel Kernel, const int num, const float alpha, float* data);
template void scal_gpu<double>(cl_kernel Kernel, const int num, const double alpha, double* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, int dim, Dtype* data, const Dtype* label){
    OCL_CHECK( clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&num) );
    OCL_CHECK( clSetKernelArg(Kernel, 1, sizeof(cl_int), (void*)&dim) );
    OCL_CHECK( clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&data) );
    OCL_CHECK( clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&label) );

    size_t Global_Work_Size[1] = {num};
    size_t Local_Work_Size[1] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL) );
}

// Explicit instantiation
template void diff_gpu<float>(cl_kernel Kernel, const int num, const int dim, float* data, const float* label);
template void diff_gpu<double>(cl_kernel Kernel, const int num, const int dim, double* data, const double* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* top_data){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);

    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template  void max_pool_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, float* top_data);
template  void max_pool_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, double* top_data);

template <typename Dtype> 
void ave_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* top_data){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel, 10,sizeof(cl_int), (void*)&pad_);
    ret |= clSetKernelArg(Kernel, 11,sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {count * 1};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void ave_pool_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, float* top_data);
template void ave_pool_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_,const int stride_,const int pad_, double* top_data);

template <typename Dtype> 
void max_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* bottom_diff ){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&top_data);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel,11, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,12, sizeof(cl_mem), (void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {count};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL));
}

template void max_pool_bp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, const float* top_data, const float* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, float* bottom_diff);
template void max_pool_bp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, const double* top_data, const double* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, double* bottom_diff );

template <typename Dtype> 
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* bottom_diff){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_int), (void*)&clnum);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_int), (void*)&channels_);
    ret |= clSetKernelArg(Kernel, 4, sizeof(cl_int), (void*)&height_);
    ret |= clSetKernelArg(Kernel, 5, sizeof(cl_int), (void*)&width_);
    ret |= clSetKernelArg(Kernel, 6, sizeof(cl_int), (void*)&pooled_height_);
    ret |= clSetKernelArg(Kernel, 7, sizeof(cl_int), (void*)&pooled_width_);
    ret |= clSetKernelArg(Kernel, 8, sizeof(cl_int), (void*)&kernel_size_);
    ret |= clSetKernelArg(Kernel, 9, sizeof(cl_int), (void*)&stride_);
    ret |= clSetKernelArg(Kernel,10, sizeof(cl_int), (void*)&pad_);
    ret |= clSetKernelArg(Kernel,11, sizeof(cl_mem), (void*)&bottom_diff);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[]={count};
    size_t uiLocal_Work_Size[]={256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL));
}

template void ave_pool_bp_gpu<float>(cl_kernel Kernel, const int count, const float* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, float* bottom_diff);
template void ave_pool_bp_gpu<double>(cl_kernel Kernel, const int count, const double* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, double* bottom_diff);

template <typename Dtype> 
void Relu_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, Dtype* top_data){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&top_data);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void Relu_fp_gpu<float>(cl_kernel Kernel, const int count, const float* bottom_data, float* top_data);
template void Relu_fp_gpu<double>(cl_kernel Kernel, const int count, const double* bottom_data, double* top_data);

template <typename Dtype> 
void Relu_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&count);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&top_diff);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&bottom_data);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&bottom_diff);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {count * 1};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void Relu_bp_gpu<float>(cl_kernel Kernel, const int count, const float* top_diff, const float* bottom_data, float* bottom_diff);
template void Relu_bp_gpu<double>(cl_kernel Kernel, const int count, const double* top_diff, const double* bottom_data, double* bottom_diff);

template <typename Dtype>
void caffe_gpu_sign(cl_kernel Kernel,const int N,  const Dtype* X, Dtype * Y ){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&N);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&X);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&Y);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {N};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_sign<float>(cl_kernel Kernel,const int N,  const float* X, float* Y );
template void caffe_gpu_sign<double>(cl_kernel Kernel,const int N,  const double* X, double* Y );

template <typename Dtype>
void caffe_gpu_div (cl_kernel Kernel, const int n, const Dtype* a, const Dtype* b, Dtype* y){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&n);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&a);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&b);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&y);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {n};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_div<float> (cl_kernel Kernel, const int n, const float* a, const float* b, float* y);
template void caffe_gpu_div<double> (cl_kernel Kernel, const int n, const double* a, const double* b, double* y);

template <typename Dtype>
void caffe_gpu_add_scalar (cl_kernel Kernel, const int n, const Dtype alpha, Dtype* y){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&n);
    ret |= clSetKernelArg(Kernel, 1, sizeof(Dtype), (void*)&alpha);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&y);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {n};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_add_scalar<float> (cl_kernel Kernel, const int n, const float alpha, float* y);
template void caffe_gpu_add_scalar<double> (cl_kernel Kernel, const int n, const double alpha, double* y);

template <typename Dtype>
void caffe_gpu_mul (cl_kernel Kernel, const int n, const Dtype* a, const Dtype* b, Dtype* y){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&n);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&a);
    ret |= clSetKernelArg(Kernel, 2, sizeof(cl_mem), (void*)&b);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&y);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {n};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_mul<float> (cl_kernel Kernel, const int n, const float* a, const float* b, float* y);
template void caffe_gpu_mul<double> (cl_kernel Kernel, const int n, const double* a, const double* b, double* y);

template <typename Dtype>
void caffe_gpu_powx (cl_kernel Kernel, const int n, const Dtype* a, const Dtype alpha, Dtype* y){
    cl_int ret;
    ret  = clSetKernelArg(Kernel, 0, sizeof(cl_int), (void*)&n);
    ret |= clSetKernelArg(Kernel, 1, sizeof(cl_mem), (void*)&a);
    ret |= clSetKernelArg(Kernel, 2, sizeof(Dtype), (void*)&alpha);
    ret |= clSetKernelArg(Kernel, 3, sizeof(cl_mem), (void*)&y);
    OCL_CHECK(ret);
    size_t Global_Work_Size[] = {n};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void caffe_gpu_powx<float> (cl_kernel Kernel, const int n, const float* a, const float alpha, float* y);
template void caffe_gpu_powx<double> (cl_kernel Kernel, const int n, const double* a, const double alpha, double* y);

template <typename Dtype>
void Dropout_fp_gpu(cl_kernel kernel, const int count, const Dtype* bottom_data, const int* MaskMem, const Dtype scale_, Dtype* top_data)
{
    cl_int ret;
    ret=clSetKernelArg(kernel,0,sizeof(cl_int),(void*)&count);
    ret|=clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&bottom_data);
    ret|=clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&MaskMem);
    ret|=clSetKernelArg(kernel,3,sizeof(cl_float),(void*)&scale_); 
    ret|=clSetKernelArg(kernel,4,sizeof(cl_mem),(void*)&top_data); 
    OCL_CHECK(ret);

    size_t Global_Work_Size[] = {count};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}

template void Dropout_fp_gpu<float>(cl_kernel kernel, const int count, const float* bottom_data, const int* MaskMem, const float scale_, float* top_data);
template void Dropout_fp_gpu<double>(cl_kernel kernel, const int count, const double* bottom_data, const int* MaskMem, const double scale_, double* top_data);

template <typename Dtype>
void Dropout_bp_gpu(cl_kernel kernel, const int count, const Dtype* top_diff, const int* MaskMem, const float threshold_, const Dtype scale_, Dtype* bottom_diff)
{
    cl_int ret;
    ret = clSetKernelArg(kernel, 0,sizeof(cl_int),  (void*)&count);
    ret |= clSetKernelArg(kernel,1,sizeof(cl_mem),  (void*)&top_diff);
    ret |= clSetKernelArg(kernel,2,sizeof(cl_mem),  (void*)&MaskMem);
    ret |= clSetKernelArg(kernel,3,sizeof(cl_int),  (void*)&threshold_); 
    ret |= clSetKernelArg(kernel,4,sizeof(cl_float),(void*)&scale_); 
    ret |= clSetKernelArg(kernel,5,sizeof(cl_mem),  (void*)&bottom_diff); 
    OCL_CHECK(ret);
   
    size_t Global_Work_Size[] = {count};
    size_t Local_Work_Size[] = {256};
    OCL_CHECK(clEnqueueNDRangeKernel(amdDevice.CommandQueue, kernel, 1, NULL, Global_Work_Size, Local_Work_Size, 0, NULL, NULL));
}
template void Dropout_bp_gpu<float>(cl_kernel kernel, const int count, const float* top_diff, const int* MaskMem, const float threshold_, const float scale_, float* bottom_diff);
template void Dropout_bp_gpu<double>(cl_kernel kernel, const int count, const double* top_diff, const int* MaskMem, const float threshold_, const double scale_, double* bottom_diff);

typedef unsigned int uint32_t;
struct array4x32 {  uint32_t v[4]; };
template <typename Dtype>
void caffe_gpu_bernoulli(cl_kernel ker_rand, int* a, const unsigned int n, Dtype inf, Dtype sup, Dtype threshold){
        static unsigned c = 0;
        unsigned nrounds = 20;
        array4x32  rndctr4;
        rndctr4.v[0] = rndctr4.v[1] = rndctr4.v[2] = rndctr4.v[3] = c++;
        cl_uint size = n / 4; //Note: for correctness, we need to make sure n is dividable by 4
        
        cl_int ret;
        ret  = clSetKernelArg(ker_rand, 0, sizeof(cl_mem),     (void*)&a);
        ret |= clSetKernelArg(ker_rand, 1, sizeof(array4x32),  (void*)&rndctr4);
        ret |= clSetKernelArg(ker_rand, 2, sizeof(cl_float),   (void*)&inf);
        ret |= clSetKernelArg(ker_rand, 3, sizeof(cl_float),   (void*)&sup);
        ret |= clSetKernelArg(ker_rand, 4, sizeof(cl_float),   (void*)&threshold);
        ret |= clSetKernelArg(ker_rand, 5, sizeof(cl_uint),    (void*)&nrounds);
        ret |= clSetKernelArg(ker_rand, 6, sizeof(cl_uint),    (void*)&size);
        OCL_CHECK(ret);

        size_t globalws[1] = {size};
        size_t localws[1] = {256};
        OCL_CHECK (clEnqueueNDRangeKernel(amdDevice.CommandQueue, ker_rand, 1, NULL, globalws, localws, 0, NULL, NULL) );
}
template void caffe_gpu_bernoulli<float>(cl_kernel kernel, int* a, const unsigned int n, float inf, float sup, float threshold);
template void caffe_gpu_bernoulli<double>(cl_kernel kernel, int* a, const unsigned int n, double inf, double sup, double threshold);


template <typename Dtype>
void opttrans(cl_kernel Kernel, const Dtype* data_im, const int im_offset, const int channels,
    const int height, const int width, Dtype* data_opt, const int opt_offset, const int optnum) {

    int num_kernels = channels * height * width * optnum;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operatiors)

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&im_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_mem),(void*)&data_opt);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&opt_offset);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&optnum);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}

template void opttrans<float>(cl_kernel Kernel, const float* data_im, const int im_offset, const int channels,
    const int height, const int width, float* data_opt, const int opt_offset, const int optnum);
template void opttrans<double>(cl_kernel Kernel, const double* data_im, const int im_offset, const int channels,
    const int height, const int width, double* data_opt, const int opt_offset, const int optnum);


}  // namespace caffe

