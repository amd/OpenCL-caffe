// Copyright 2014 AMD DNN contributors.

#ifndef _CAFFE_UTIL_OCL_WRAPPER_HPP_
#define _CAFFE_UTIL_OCL_WRAPPER_HPP_

namespace caffe {

typedef unsigned int uint32_t;
template <typename Dtype>
void caffe_gpu_bernoulli(cl_kernel ker_rand, int* a, const unsigned int n, Dtype inf, Dtype sup, Dtype threshold);

template <typename Dtype>
void transform_gpu(cl_kernel Kernel, Dtype* src, Dtype* dst, const int top_offset, const int N_, const int M_, const int packing_num);

template <typename Dtype>
void opttrans(cl_kernel Kernel, const Dtype* data_im, const int im_offset, const int channels,
    const int height, const int width, Dtype* data_opt, const int opt_offset, const int optnum);

template <typename Dtype>
void get_max_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* bottom_data, Dtype* scale_data);

template <typename Dtype>
void exp_gpu(cl_kernel Kernel, const int num, const Dtype* data, Dtype* out);

template <typename Dtype>
void softmax_div_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* scale, Dtype* data);

template <typename Dtype>
Dtype softmax_gpu(cl_kernel Kernel, const int num, const int dim, const Dtype* prob_data, const Dtype* label, cl_mem d_loss);

template <typename Dtype>
void scal_gpu(cl_kernel Kernel, const int num, const Dtype alpha, Dtype* data);

template <typename Dtype>
void diff_gpu(cl_kernel Kernel, const int num, const int dim, Dtype* data, const Dtype* label);

template <typename Dtype>
void max_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* top_data);

template <typename Dtype>
void MaxPoolForward(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_h_, const int kernel_w_, const int stride_h_, const int stride_w_, const int pad_h_, const int pad_w_, Dtype* top_data, int* mask, Dtype* top_mask);

template <typename Dtype>
void MaxPoolBackward(cl_kernel kernel, const int nthreads, const Dtype* const top_diff, const int* const mask, const Dtype* const top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, Dtype* const bottom_diff);

template <typename Dtype>
void AvePoolBackward(cl_kernel kernel, const int nthreads, const Dtype* const top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, Dtype* const bottom_diff);

template <typename Dtype>
 void StoPoolBackward(cl_kernel kernel, const int nthreads, const Dtype* const rand_idx, const Dtype* const top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, Dtype* const bottom_diff);

template <typename Dtype>
void ave_pool_fp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* top_data);

template <typename Dtype>
void AvePoolForward(cl_kernel Kernel,const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_,  const int kernel_h_, const int kernel_w_, const int stride_h_, const int stride_w_, const int pad_h_, const int pad_w_, Dtype* top_data);

template <typename Dtype>
void StoPoolForwardTrain(cl_kernel Kernel,const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_,  const int kernel_h_, const int kernel_w_, const int stride_h_, const int stride_w_, Dtype* idx_data, Dtype* top_data);

template <typename Dtype>
void StoPoolForwardTest(cl_kernel Kernel,const int count, const Dtype* bottom_data, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_,  const int kernel_h_, const int kernel_w_, const int stride_h_, const int stride_w_, Dtype* top_data);

template <typename Dtype>
void max_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* bottom_data, const Dtype* top_data, const Dtype* top_diff, const int clnum, const int channels_, const int height_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, Dtype* bottom_diff );

template <typename Dtype>
void ave_pool_bp_gpu(cl_kernel Kernel, const int count, const Dtype* top_diff, const int clnum, const int channels_, const int intheight_, const int width_, const int pooled_height_, const int pooled_width_, const int kernel_size_, const int stride_, const int pad_, Dtype* bottom_diff);

template <typename Dtype>
void ReLUForward(const int count, const Dtype* bottom_data, Dtype* top_data, Dtype negative_slope);

template <typename Dtype>
void ReLUBackward(const int count, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff, Dtype negative_slope);

template <typename Dtype>
void caffe_gpu_div (cl_kernel Kernel, const int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void DropoutForward(cl_kernel kernel, const int count, const Dtype* bottom_data, const int* MaskMem, const Dtype scale_, Dtype *top_data);

template <typename Dtype>
void DropoutBackward(cl_kernel kernel, const int count, const Dtype* top_diff, const int* MaskMem, const float threshold_, const Dtype scale_, Dtype* bottom_diff);

template <typename Dtype>
void caffe_gpu_bernoulli(cl_kernel ker_rand, int* a, const unsigned int n, Dtype inf, Dtype sup, Dtype threshold);

template <typename Dtype>
void caffe_gpu_sign(cl_kernel Kernel,const int N, const Dtype* X, Dtype * Y );

template <typename Dtype>
void kernel_channel_max(cl_kernel Kernel, const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out);

template <typename Dtype>
void kernel_channel_subtract(cl_kernel Kernel, const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data);

template <typename Dtype>
void kernel_exp(cl_kernel Kernel, const int count, const Dtype* data, Dtype* out);

template <typename Dtype>
void kernel_channel_sum(cl_kernel Kernel, const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum);

template <typename Dtype>
void kernel_channel_div(cl_kernel Kernel, const int count, const int num, const int channels, const int spatial_dim, const Dtype* channel_sum, Dtype* data);

template <typename Dtype>
void kernel_channel_dot(cl_kernel Kernel, const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot);

template <typename Dtype>
void SoftmaxLossForwardGPU(cl_kernel Kernel, const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts);

template <typename Dtype>
void SoftmaxLossBackwardGPU(cl_kernel Kernel, const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts);

template <typename Dtype>
void caffe_gpu_add(cl_kernel Kernel, const int n, const Dtype* in1, const Dtype* in2, Dtype* y);

template <typename Dtype>
void caffe_gpu_add_scalar(cl_kernel Kernel, const int n, const Dtype alpha, Dtype* top_data);

template <typename Dtype>
void LRNFillScale(cl_kernel LFSkernel, const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale);

template <typename Dtype>
void LRNComputeOutput(cl_kernel LCOkernel, int nthreads, const Dtype* in,
     Dtype* scale, Dtype negative_beta, Dtype* out);

template <typename Dtype>
void LRNComputeDiff(cl_kernel LCDkernel, const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff);
template <typename Dtype>
void caffe_gpu_powx (cl_kernel Kernel, const int n, const Dtype* a, const Dtype alpha, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul (cl_kernel Kernel, const int n, const Dtype* a, const Dtype* b, Dtype* y);
}
#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
  // namespace caffe
