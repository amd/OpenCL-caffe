template <class Dtype>
__kernel void MaxForward(const int nthreads, __global const Dtype* bottom_data_a,
    __global const Dtype* bottom_data_b, const int blob_idx, __global Dtype* top_data,
    __global int* mask) {
    int index = get_global_id(0);
    if(index < nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}
template __attribute__((mangled_name(MaxForward_float))) __kernel void MaxForward(const int nthreads, __global const float* bottom_data_a,
    __global const float* bottom_data_b, const int blob_idx, __global float* top_data,
    __global int* mask);
template __attribute__((mangled_name(MaxForward_double))) __kernel void MaxForward(const int nthreads, __global const double* bottom_data_a,
    __global const double* bottom_data_b, const int blob_idx, __global double* top_data,
    __global int* mask);

template <class Dtype>
__kernel void MaxBackward(const int nthreads, __global const Dtype* top_diff,
    const int blob_idx, __global const int* mask, __global Dtype* bottom_diff) {
    int index = get_global_id(0);
    if(index < nthreads) {
        Dtype gradient = 0;
        if (mask[index] == blob_idx) {
            gradient += top_diff[index];
        }
        bottom_diff[index] = gradient;
    }
}
template __attribute__((mangled_name(MaxBackward_float))) __kernel void MaxBackward(const int nthreads, __global const float* top_diff,
    const int blob_idx, __global const int* mask, __global float* bottom_diff);
template __attribute__((mangled_name(MaxBackward_double))) __kernel void MaxBackward(const int nthreads, __global const double* top_diff,
    const int blob_idx, __global const int* mask, __global double* bottom_diff);
