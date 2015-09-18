#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer < Dtype > ::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(count, bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i * channels), diff_.cpu_data() + (i * channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max(margin - sqrt(dist_sq_.cpu_data()[i]), 0.0);
        loss += dist * dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(channels, alpha, diff_.cpu_data() + (j * channels),
              Dtype(0.0), bout + (j * channels));
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4));
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(channels, beta, diff_.cpu_data() + (j * channels),
                Dtype(0.0), bout + (j * channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j * channels));
          }
        }
      }
    }
  }
}

// begin: code modified for OpenCL port
#ifndef CPU_ONLY
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_sub(count, bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(count, diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2), diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->channels(),
      Dtype(1.0), diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(), Dtype(0.0), dist_sq_.mutable_gpu_data()); // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max(margin - sqrt(dist_sq_.cpu_data()[i]), 0.0);
        loss += dist * dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[0]->count();
      const int channels = bottom[0]->channels();
      Dtype margin = this->layer_param_.contrastive_loss_param().margin();
      const bool legacy_version =
          this->layer_param_.contrastive_loss_param().legacy_version();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0]
          / static_cast<Dtype>(bottom[0]->num());
      // NOLINT_NEXT_LINE(whitespace/operators)
      CLLBackward(count, channels, margin, legacy_version, alpha,
          bottom[2]->gpu_data(),  // pair similarity 0 or 1
          diff_.gpu_data(),  // the cached eltwise difference between a and b
          dist_sq_.gpu_data(),  // the cached square distance between a and b
          bottom[i]->mutable_gpu_diff());
    }
  }
}
// end: code modified for OpenCL port

#else
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS (ContrastiveLossLayer);
REGISTER_LAYER_CLASS (ContrastiveLoss);

}  // namespace caffe
