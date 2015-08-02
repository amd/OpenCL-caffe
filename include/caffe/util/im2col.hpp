#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void col2im_gpu(cl_kernel Kernel, const Dtype* data_col, const int col_offset,
    const int height, const int width, const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im, const int img_offset);

template <typename Dtype>
void im2col_gpu(cl_kernel Kernel,  const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col, const int col_offset);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

template <typename Dtype>
void im2col_gpu(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset);

template <typename Dtype>
void im2col_16_gpu(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset);

template <typename Dtype>
void im2col_opt_gpu(cl_kernel Kernel, const Dtype* data_im, const int img_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, const int col_offset, int optnum);

template <typename Dtype>
void col2im_gpu(cl_kernel Kernel, const Dtype* data_col, const int col_offset, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, Dtype* data_im, const int img_offset);

template <typename Dtype>
void col2im_gpu_opt(cl_kernel Kernel, const Dtype* data_col, const int col_offset, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im, const int img_offset, int optnum);

template <typename Dtype>
void col2im_gpu_ocl(cl_mem data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im, cl_kernel Kernel);

template <typename Dtype>
void im2col_gpu_ocl(cl_mem data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col, cl_kernel Kernel);
}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
