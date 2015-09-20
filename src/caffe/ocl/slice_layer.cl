template <class Dtype>
__kernel void Slice(const int nthreads, __global const Dtype* in_data,
    const int forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, __global Dtype* out_data) {
    int index = get_global_id(0);
    if (index < nthreads) {
        const int total_slice_size = slice_size * top_slice_axis;
        const int slice_num = index / total_slice_size;
        const int slice_index = index % total_slice_size;
        const int bottom_index = slice_index +
            (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
        if (forward) {
            out_data[index] = in_data[bottom_index];
        } else {
            out_data[bottom_index] = in_data[index];
        }
  }
}

template __attribute__ ((mangled_name(Slice_float))) __kernel void Slice(const int nthreads, __global const float* in_data,
    const int forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, __global float* out_data);
template __attribute__ ((mangled_name(Slice_double))) __kernel void Slice(const int nthreads, __global const double* in_data,
    const int forward, const int num_slices, const int slice_size,
    const int bottom_slice_axis, const int top_slice_axis,
    const int offset_slice_axis, __global double* out_data);
