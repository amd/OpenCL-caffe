template <class T>
__kernel void Concat(const int nthreads, __global const T* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, __global T* out_data) {
    int index = get_global_id(0);
    if(index < nthreads) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num = index / total_concat_size;
        const int concat_index = index % total_concat_size;
        const int top_index = concat_index +
            (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
        if (forward) {
            out_data[top_index] = in_data[index];
        } else {
            out_data[index] = in_data[top_index];
        }
    }
}

template __attribute__((mangled_name(Concat_float))) __kernel void  Concat(const int nthreads, __global const float* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, __global float* out_data);
template __attribute__((mangled_name(Concat_double))) __kernel void  Concat(const int nthreads, __global const double* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, __global double* out_data);
