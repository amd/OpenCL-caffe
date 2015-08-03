#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);


template <typename Dtype>
void col2im_gpu_opt(cl_kernel Kernel, const Dtype* data_col, const int col_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im, const int img_offset, int optnum){
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&col_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&img_offset);
    ret|=clSetKernelArg(Kernel,13,sizeof(cl_int),(void*)&optnum);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}

template void col2im_gpu_opt<float>(cl_kernel kernel, const float* data_col, const int col_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_im, const int img_offset, int optnum);
template void col2im_gpu_opt<double>(cl_kernel kernel, const double* data_col, const int col_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_im, const int img_offset, int optnum);

//cannot use now, need to modify kernel.
template <typename Dtype>
void im2col_gpu(cl_kernel Kernel,  const Dtype* data_im, const int img_offset, const int channels, 
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col, const int col_offset)
{
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&img_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&kernel_h);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&kernel_w);

    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pad_h);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&pad_w);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&stride_h);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&stride_w);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,13,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,14,sizeof(cl_int),(void*)&col_offset);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
    clFinish(amdDevice.CommandQueue);

}

template void im2col_gpu<float>(cl_kernel Kernel,  const float* data_im, const int img_offset, const int channels,       
    				const int height, const int width, const int kernel_h, const int kernel_w,
    				const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    				float* data_col, const int col_offset);
template void im2col_gpu<double>(cl_kernel Kernel,  const double* data_im, const int img_offset, const int channels,       
    				const int height, const int width, const int kernel_h, const int kernel_w,
    				const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    				double* data_col, const int col_offset);

//cannot use now, need to modify kernel
template <typename Dtype>
void col2im_gpu(cl_kernel Kernel, const Dtype* data_col, const int col_offset,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im, const int img_offset)
{
    int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
    int num_kernels = channels * height * width;
    
    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&col_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&patch_h);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&patch_w);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&pad_h);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&pad_w);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&stride_h);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_int),(void*)&stride_w);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,13,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,14,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,15,sizeof(cl_int),(void*)&img_offset);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}

template void col2im_gpu<float>(cl_kernel Kernel, const float* data_col, const int col_offset,
    				const int height, const int width, const int channels,
    				const int patch_h, const int patch_w, const int pad_h, const int pad_w,
    				const int stride_h, const int stride_w, float* data_im, const int img_offset);
template void col2im_gpu<double>(cl_kernel Kernel, const double* data_col, const int col_offset,
    				const int height, const int width, const int channels,
    				const int patch_h, const int patch_w,
    				const int pad_h, const int pad_w,const int stride_h, const int stride_w,
    				double* data_im, const int img_offset);

template <typename Dtype>
void im2col_gpu(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    
    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&img_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_int),(void*)&col_offset);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
    clFinish(amdDevice.CommandQueue);
}

template void im2col_gpu<float>(cl_kernel Kernel, const float* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col, const int col_offset);
template void im2col_gpu<double>(cl_kernel Kernel, const double* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col, const int col_offset);

template <typename Dtype>
void im2col_16_gpu(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = 16 * channels * height_col * width_col;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&img_offset);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&col_offset);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256 - 256 % width_col};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}

template void im2col_16_gpu<float>(cl_kernel Kernel, const float* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col, const int col_offset);
template void im2col_16_gpu<double>(cl_kernel Kernel, const double* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col, const int col_offset);

template <typename Dtype>
void im2col_gpu_opt(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset, int optnum) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = optnum * channels * height_col * width_col;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&img_offset);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&col_offset);
    ret|=clSetKernelArg(Kernel,13,sizeof(cl_int),(void*)&optnum);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256 - 256 % width_col};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}

template void im2col_gpu_opt<float>(cl_kernel Kernel, const float* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col, const int col_offset, int optnum);
template void im2col_gpu_opt<double>(cl_kernel Kernel, const double* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col, const int col_offset,  int optnum);

template <typename Dtype>
void col2im_gpu(cl_kernel Kernel, const Dtype* data_col, const int col_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im, const int img_offset) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operatiors)

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&col_offset);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,11,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,12,sizeof(cl_int),(void*)&img_offset);
    OCL_CHECK(ret);

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {256};
    OCL_CHECK( clEnqueueNDRangeKernel(amdDevice.CommandQueue, Kernel, 1, NULL, uiGlobal_Work_Size, uiLocal_Work_Size, 0, NULL, NULL) );
}


template void col2im_gpu<float>(cl_kernel Kernel, const float* data_col, const int col_offset, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im, const int img_offset);
template void col2im_gpu<double>(cl_kernel Kernel, const double* data_col, const int col_offset, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im, const int img_offset);

template <typename Dtype>
void im2col_gpu_ocl(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, cl_kernel Kernel) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,1,sizeof(cl_mem),(void*)&data_im);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&width_col);
    OCL_CHECK( clSetKernelArg(Kernel,9,sizeof(cl_mem),(void*)&data_col) );

    if(ret!=CL_SUCCESS){
        fprintf(stderr,"Failed to Set Args\n");
    }

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {64};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL);
    if(CL_SUCCESS!=iStatus){
        fprintf(stderr,"Failed to enqueue kernel\n");
    }
}

template void im2col_gpu_ocl<float>(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col, cl_kernel Kernel);
template void im2col_gpu_ocl<double>(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col, cl_kernel Kernel);

template <typename Dtype>
void col2im_gpu_ocl(cl_mem data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im, cl_kernel Kernel) {

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operatiors)

    cl_int ret;
    ret=clSetKernelArg(Kernel,0,sizeof(cl_int),(void*)&num_kernels);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_mem),(void*)&data_col);
    ret|=clSetKernelArg(Kernel,2,sizeof(cl_int),(void*)&height);
    ret|=clSetKernelArg(Kernel,3,sizeof(cl_int),(void*)&width);
    ret|=clSetKernelArg(Kernel,4,sizeof(cl_int),(void*)&channels);
    ret|=clSetKernelArg(Kernel,5,sizeof(cl_int),(void*)&ksize);
    ret|=clSetKernelArg(Kernel,6,sizeof(cl_int),(void*)&pad);
    ret|=clSetKernelArg(Kernel,7,sizeof(cl_int),(void*)&stride);
    ret|=clSetKernelArg(Kernel,8,sizeof(cl_int),(void*)&height_col);
    ret|=clSetKernelArg(Kernel,9,sizeof(cl_int),(void*)&width_col);
    ret|=clSetKernelArg(Kernel,10,sizeof(cl_mem),(void*)&data_im);

    if(ret!=CL_SUCCESS){
        fprintf(stderr,"Failed to Set Args\n");
    }

    size_t uiGlobal_Work_Size[] = {num_kernels};
    size_t uiLocal_Work_Size[] = {64};
    cl_int iStatus = clEnqueueNDRangeKernel(amdDevice.CommandQueue,Kernel,1,NULL,uiGlobal_Work_Size,uiLocal_Work_Size,0,NULL,NULL);
    if(CL_SUCCESS!=iStatus){
        fprintf(stderr,"Failed to enqueue kernel\n");
    }
}


template void col2im_gpu_ocl<float>(cl_mem data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im, cl_kernel Kernel);
template void col2im_gpu_ocl<double>(cl_mem data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im, cl_kernel Kernel);

}  // namespace caffe
