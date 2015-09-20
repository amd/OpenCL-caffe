#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY  // CPU-only Caffe.

#include <vector>

// Stub out GPU calls as unavailable.

#define NO_GPU LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode."

#define STUB_GPU(classname) \
template <typename Dtype> \
void classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    const vector<Blob<Dtype>*>& top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    const vector<Blob<Dtype>*>& bottom) { NO_GPU; } \

#else  // Normal GPU + CPU Caffe.

#ifdef USE_CUDNN  // cuDNN acceleration library.
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {
}  // namespace caffe

#endif  // CPU_ONLY

#endif  // CAFFE_UTIL_DEVICE_ALTERNATE_H_
