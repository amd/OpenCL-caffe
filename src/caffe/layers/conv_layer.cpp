#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }

  // CHECK_BLOB_DATA(top[0],20, "top[0]");
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }

}

// begin: code modified for OpenCL port
#ifndef CPU_ONLY
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  if (!this->is_1x1_ && use_packing_scheme && global_packing_N > 1)
    Forward_gpu_batched(bottom, top);
  else
    Forward_gpu_org(bottom, top);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!this->is_1x1_ && use_packing_scheme && global_packing_N > 1)
    Backward_gpu_batched(top, propagate_down, bottom);
  else
    Backward_gpu_org(top, propagate_down, bottom);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu_batched(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    //CHECK_BLOB_DATA(bottom[i],10,"bottom");

    Dtype* top_data = top[i]->mutable_gpu_data();
    this->opt_num2 = global_packing_N;
    this->weight_offset_ = this->M_ * this->K_;
    for (int n = 0; n < this->num_; n += this->opt_num2) {
      this->opt_num2 =
          this->opt_num2 > (this->num_ - n) ? (this->num_ - n) : this->opt_num2;
      //intermediate variables to pass offset
      this->top_offset_opt = this->M_ * this->N_ * this->opt_num2;
      this->top_offset_ = top[i]->offset(n);
      this->col_offset_ = this->K_ * this->N_ * this->opt_num2;
      this->bottom_offset_ = bottom[i]->offset(n);
      this->forward_gpu_gemm_opt(bottom_data, weight, top_data);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias_opt(top_data, bias);
      }
    }
  }

  //CHECK_BLOB_DATA(this->blobs_[0],20, "weights");
  //CHECK_BLOB_DATA(top[0],20, "top[0]");

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu_org(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      //two intermediate variables to pass offset
      this->bottom_offset_ = bottom[i]->offset(n);
      this->top_offset_ = top[i]->offset(n);
      this->col_offset_ = this->K_ * this->N_;
      this->forward_gpu_gemm(bottom_data, weight, top_data);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, bias);
      }
    }
  }

  // CHECK_BLOB_DATA(this->blobs_[0],20, "weights");
  //CHECK_BLOB_DATA(top[0],20, "top[0]");
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu_batched(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->top_offset_ = top[i]->offset(n);
        this->backward_gpu_bias(bias_diff, top_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      this->weight_offset_ = this->M_ * this->K_;
      this->opt_num2 = global_packing_N;
      for (int n = 0; n < this->num_; n += this->opt_num2) {
        this->opt_num2 =
            this->opt_num2 > (this->num_ - n) ?
                (this->num_ - n) : this->opt_num2;
        this->top_offset_ = top[i]->offset(n);
        this->bottom_offset_ = bottom[i]->offset(n);
        this->col_offset_ = this->K_ * (this->N_ * this->opt_num2);
        this->top_offset_opt = this->M_ * (this->N_ * this->opt_num2);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_opt(bottom_data, top_diff, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_opt(top_diff, weight, bottom_diff);
        }
      }
    }
  }
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu_org(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        //
        this->top_offset_ = top[i]->offset(n);
        this->bottom_offset_ = bottom[i]->offset(n);
        this->col_offset_ = this->K_ * this->N_;
        this->backward_gpu_bias(bias_diff, top_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->top_offset_ = top[i]->offset(n);
        this->bottom_offset_ = bottom[i]->offset(n);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, top_diff, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff, weight, bottom_diff);
        }
      }
    }
  }

}
// end: code modified for OpenCL port

#else
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS (ConvolutionLayer);

}  // namespace caffe
