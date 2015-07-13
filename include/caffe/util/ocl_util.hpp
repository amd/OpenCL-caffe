// Copyright 2014 AMD DNN contributors.

#ifndef _CAFFE_UTIL_OCL_UTIL_HPP_
#define _CAFFE_UTIL_OCL_UTIL_HPP_

namespace caffe {

template <typename Dtype>
void ocl_memset(cl_kernel Kernel, Dtype* buffer, const Dtype value, const int count);

void ocl_memset(cl_kernel Kernel, cl_mem buffer, const int value, const int count);

void eventCallback(cl_event event, cl_int event_status, void * user_data);
}  // namespace caffe

#endif  // CAFFE_UTIL_OCL_UTIL_HPP_
