template <class T>
__kernel void SoftmaxLossForwardGPU(const int nthreads,
          __global T* prob_data, __global T* label,__global T* loss,
          int num, int dim, int spatial_dim,
          bool has_ignore_label_, int ignore_label_,
          __global T* counts) {
    int index = get_global_id(0);
    if(index < nthreads) {
        const int n = index / spatial_dim;
        const int s = index % spatial_dim;
        const int label_value = static_cast<int>(label[n * spatial_dim + s]);
        if (has_ignore_label_ && label_value == ignore_label_) {
           loss[index] = 0;
           counts[index] = 0;
        } else {
           loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      T(FLT_MIN)));
        counts[index] = 1;
    }
  }
}

template __attribute__ ((mangled_name(softmax_loss_fp_float))) __kernel void SoftmaxLossForwardGPU(int nthreads,
          __global float* prob_data, __global float* label,__global float* loss,
          int num, int dim, int spatial_dim,
          bool has_ignore_label_, int ignore_label_,
          __global float* counts);
template __attribute__ ((mangled_name(softmax_loss_fp_double))) __kernel void SoftmaxLossForwardGPU(int nthreads,
          __global double* prob_data, __global double* label,__global double* loss,
          int num, int dim, int spatial_dim,
          bool has_ignore_label_, int ignore_label_,
          __global double* counts);

template <class T>
__kernel void SoftmaxLossBackwardGPU(int nthreads, __global T* top,
          __global T* label,__global T* bottom_diff, int num, int dim,
          int spatial_dim, bool has_ignore_label_,
          int ignore_label_, T* counts) {
    const int channels = dim / spatial_dim;
   int index  = get_global_id(0);
   if(index <  nthreads) {
       const int n = index / spatial_dim;
       const int s = index % spatial_dim;
       const int label_value = static_cast<int>(label[n * spatial_dim + s]);

      if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < channels; ++c) {
              bottom_diff[n * dim + c * spatial_dim + s] = 0;
          }
          counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
template __attribute__ ((mangled_name(softmax_loss_bp_float))) __kernel void SoftmaxLossBackwardGPU(int nthreads, __global float* top,
          __global float* label,__global float* bottom_diff, int num, int dim,
          int spatial_dim, bool has_ignore_label_,
          int ignore_label_, float* counts);

template __attribute__ ((mangled_name(softmax_loss_bp_double)))  __kernel void SoftmaxLossBackwardGPU(int nthreads, __global double* top,
          __global double* label,__global double* bottom_diff, int num, int dim,
          int spatial_dim, bool has_ignore_label_,
          int ignore_label_, double* counts);
