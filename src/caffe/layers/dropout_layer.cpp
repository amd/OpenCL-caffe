// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::ocl_setup(int bottom_count){
    //create OpenCL related cl_mem objects and kernels
    //if(Caffe::mode() == Caffe::GPU){
    cl_int _err;
    ocl_Kernel_Fwd = clCreateKernel(amdDevice.Program,"DropoutForwardfloat",&_err);
    ocl_Kernel_Bwd = clCreateKernel(amdDevice.Program,"DropoutBackwardfloat",&_err);
    rng_kernel = clCreateKernel(amdDevice.Program,"RNGBernoulliFloat",&_err);
    OCL_CHECK(_err);
    MaskMem = clCreateBuffer(amdDevice.Context, CL_MEM_READ_WRITE, bottom_count*sizeof(int), NULL, NULL);
}

template <typename Dtype>
DropoutLayer<Dtype>::~DropoutLayer(){
   OCL_CHECK( clReleaseMemObject(MaskMem) );
   OCL_CHECK( clReleaseKernel(ocl_Kernel_Fwd) );
   OCL_CHECK( clReleaseKernel(ocl_Kernel_Bwd) );
   OCL_CHECK( clReleaseKernel(rng_kernel) );
}


template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  ocl_setup(bottom[0]->count());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
//    caffe_gpu_rng_uniform(count, mask);
 
     caffe_gpu_bernoulli(rng_kernel, (int*)MaskMem, count, (Dtype)0., (Dtype)1., threshold_);
    Dropout_fp_gpu(ocl_Kernel_Fwd, count, bottom_data, (int*)MaskMem, (Dtype)scale_, top_data);

    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
//    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //      count, bottom_data, mask, uint_thres_, scale_, top_data);
   // CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_gpu_copy(count, bottom_data, top_data);
  }
}


template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
     // DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
       // CAFFE_CUDA_NUM_THREADS>>>(
         // count, top_diff, mask, uint_thres_, scale_, bottom_diff);
    //  CUDA_POST_KERNEL_CHECK;
       Dropout_bp_gpu(ocl_Kernel_Bwd, count, top_diff, (int*)MaskMem, uint_thres_ , (Dtype)scale_, bottom_diff);
    } else {
      caffe_gpu_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
