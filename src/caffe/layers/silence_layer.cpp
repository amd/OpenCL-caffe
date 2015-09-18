#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_data());
    }
  }
}

#ifndef CPU_ONLY
// begin: code modified for OpenCL port
template <typename Dtype>
void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Do nothing.
}

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
          bottom[i]->mutable_gpu_data());
    }
  }
}
// end: code modified for OpenCL port

#else
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS (SilenceLayer);
REGISTER_LAYER_CLASS (Silence);

}  // namespace caffe
