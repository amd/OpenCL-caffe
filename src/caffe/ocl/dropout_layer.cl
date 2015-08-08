template <class T>
__kernel void DropoutForward(const int n, __global T *in, __global const int* mask, const T scale, __global T *out){
    int index = get_global_id(0);
    if (index < n)
        out[index] = in[index] * scale * mask[index];
}
template __attribute__((mangled_name(DropoutForward_float))) __kernel void DropoutForward(const int n, __global float* in,  __global const int* mask, const float scale, __global float* out); 
template __attribute__((mangled_name(DropoutForward_double))) __kernel void DropoutForward(const int n, __global double* in, __global const int* mask, const double scale, __global double* out);


template <class T>
__kernel void DropoutBackward(const int n, __global T *in_diff, __global const int *mask, const int unsigned threshold, const T scale, __global T *out_diff){
    int index = get_global_id(0);
    if (index < n)
        out_diff[index] = in_diff[index] * scale * mask[index];
}
template __attribute__((mangled_name(DropoutBackward_float))) __kernel void DropoutBackward(const int n, __global float* in_diff,  __global const int* mask, const unsigned int threshold, const float scale, __global float* out_diff); 
template __attribute__((mangled_name(DropoutBackward_double))) __kernel void DropoutBackward(const int n, __global double* in_diff, __global const int* mask, const unsigned int threshold, const double scale, __global double* out_diff);
